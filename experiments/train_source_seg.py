"""Train the shared query decoder with joint NVS and panoptic objectives."""

import argparse
import importlib
import logging
import os
import sys
import threading
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

import wandb
import torch
import numpy as np
from rich import print
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import add_config_args, load_config, resolve_ckpt
from train.distributed import init_distributed, init_run

from train.utils import create_lr_scheduler, print_rank0, visualize_intermediate_results
from train.dataset import scannet_re10k_collate_fn
from train.scannet_classes import PANOPTIC_COLOR_PALLETE, STUFF_CLASSES
from easydict import EasyDict as edict
from omegaconf import OmegaConf

log = logging.getLogger(__name__)

STUFF_IDS_1BASED = [s + 1 for s in STUFF_CLASSES]
LABEL_IDS_TO_FUSE = set(STUFF_IDS_1BASED)

# Build color LUT once: index = class_id (1-based), value = RGB
_COLOR_LUT = np.zeros((22, 3), dtype=np.uint8)
for cid, rgb in PANOPTIC_COLOR_PALLETE.items():
    if cid < 22:
        _COLOR_LUT[cid] = rgb


def _seg_vis(seg_output, source_images, processor, image_size=224):
    """Create panoptic overlay on all source views (batch=0).

    Returns (V_i*H, W*2, 3) uint8 grid: [source_rgb | panoptic_overlay] per view.
    Unified queries produce consistent instance IDs across views.
    """
    batch_size = seg_output.class_queries_logits.shape[0]
    results = processor.post_process_panoptic_segmentation(
        outputs=seg_output,
        threshold=0.5,
        mask_threshold=0.5,
        overlap_mask_area_threshold=0.8,
        label_ids_to_fuse=LABEL_IDS_TO_FUSE,
        target_sizes=[(image_size, image_size)] * batch_size,
    )
    seg_maps = results[0]["segmentation"]  # [V_i, H, W]
    segments_info = results[0]["segments_info"]
    V_i = seg_maps.shape[0]

    rows = []
    for v in range(V_i):
        seg_map = seg_maps[v].cpu().numpy()

        # Build semantic color map
        sem = np.zeros_like(seg_map, dtype=np.int64)
        for info in segments_info:
            sem[seg_map == info["id"]] = info["label_id"] + 1
        color = _COLOR_LUT[sem]

        # Source RGB
        src = source_images[0, v].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
        src = (src * 255).astype(np.uint8)

        overlay = (src.astype(np.float32) * 0.4 + color.astype(np.float32) * 0.6).astype(np.uint8)
        rows.append(np.concatenate([src, overlay], axis=1))

    return np.concatenate(rows, axis=0)


# ---------------------------------------------------------------------------
# Optimizer with per-module LR scaling
# ---------------------------------------------------------------------------

def create_seg_source_optimizer(model, config):
    """Create optimizer with per-module LR scaling.

    Groups:
    - ``nvs.*``: lr × nvs_lr_scale (NVS backbone)
    - ``source_seg.*``: lr × seg_lr_scale (seg modules)
    """
    base_lr = config.training.lr
    weight_decay = config.training.weight_decay
    betas = (config.training.beta1, config.training.beta2)

    seg_cfg = config.segmentation
    nvs_lr_scale = seg_cfg.get("nvs_lr_scale", 1.0)
    seg_lr_scale = seg_cfg.get("seg_lr_scale", 1.0)

    param_groups = []
    all_param_dict = {}
    optimized_param_dict = {}

    for name, param in model.named_parameters():
        all_param_dict[name] = param
        if not param.requires_grad:
            continue
        optimized_param_dict[name] = param

        if name.startswith("nvs."):
            lr_scale = nvs_lr_scale
        elif name.startswith("source_seg."):
            lr_scale = seg_lr_scale
        else:
            lr_scale = 1.0

        wd = weight_decay if param.dim() > 1 else 0.0
        param_groups.append({
            "params": [param],
            "lr": base_lr * lr_scale,
            "weight_decay": wd,
            "name": name,
        })

    fused = config.training.get("fused_optimizer", False)
    optimizer = torch.optim.AdamW(param_groups, lr=base_lr, betas=betas, fused=fused)

    if dist.get_rank() == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in optimized_param_dict.values())
        print(f"Total parameters: {total_params:,}, Trainable: {trainable_params:,}")
        for prefix in ["nvs.", "source_seg.adapter.", "source_seg.mask2former.", "source_seg.feat_proj.", "dino_encoder."]:
            count = sum(p.numel() for n, p in optimized_param_dict.items() if n.startswith(prefix))
            if count > 0:
                print(f"  {prefix}: {count:,} params")

    return optimizer, optimized_param_dict, all_param_dict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

_ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
add_config_args(_ap, default="configs/train_source_seg.yaml")
_args = _ap.parse_args()
config = edict(OmegaConf.to_container(load_config(_args.config, _args.opts), resolve=True))
if not config.training.get("scannet_root"):
    config.training.scannet_root = os.environ.get("SCANNET_ROOT", "data/scannet")

os.environ["OMP_NUM_THREADS"] = str(config.training.get("num_threads", 1))

ddp_info = init_distributed(seed=777)
dist.barrier()

if ddp_info.is_main_process:
    init_run(config)
dist.barrier()

# TF32
torch.backends.cuda.matmul.allow_tf32 = config.training.use_tf32
torch.backends.cudnn.allow_tf32 = config.training.use_tf32
amp_dtype_mapping = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "tf32": torch.float32,
}

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

dataset_name = config.training.get("dataset_name", "src.data.scannet_re10k.Dataset")
module, class_name = dataset_name.rsplit(".", 1)
DatasetClass = importlib.import_module(module).__dict__[class_name]

dataset = DatasetClass(config)
batch_size_per_gpu = config.training.batch_size_per_gpu

# Use custom collate_fn that handles ragged mask/class labels
has_panoptic = config.get("segmentation", {}).get("load_panoptic", False)
collate_fn = scannet_re10k_collate_fn if has_panoptic else None

datasampler = DistributedSampler(dataset)
dataloader = DataLoader(
    dataset,
    batch_size=batch_size_per_gpu,
    shuffle=False,
    num_workers=config.training.num_workers,
    persistent_workers=True,
    pin_memory=False,
    drop_last=True,
    prefetch_factor=config.training.prefetch_factor,
    sampler=datasampler,
    collate_fn=collate_fn,
)
dataloader_iter = iter(dataloader)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

model_name = config.model.class_name
module, class_name = model_name.rsplit(".", 1)
ModelClass = importlib.import_module(module).__dict__[class_name]
model = ModelClass(config).to(ddp_info.device)

# Load checkpoint — auto-detect full resume vs fresh init
resume_ckpt = resolve_ckpt(config.training.get("resume_ckpt", ""))
_is_full_ckpt = False  # will be set True if checkpoint has optimizer state

if resume_ckpt and os.path.exists(resume_ckpt):
    ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model", ckpt)
    cleaned = {k.removeprefix("module."): v for k, v in state_dict.items()}

    has_seg_keys = any(k.startswith("source_seg.") for k in cleaned)
    has_optimizer = "optimizer" in ckpt

    if has_seg_keys:
        # Full checkpoint (from previous train_seg_source run)
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        print_rank0(f"Loaded full checkpoint from {resume_ckpt}")
        print_rank0(f"  Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        _is_full_ckpt = has_optimizer
    else:
        # NVS-only checkpoint — load NVS and seg separately
        model.load_nvs_checkpoint(resume_ckpt)
        seg_ckpt_path = resolve_ckpt(config.get("segmentation", {}).get("pretrained_seg_ckpt", ""))
        if seg_ckpt_path and os.path.exists(seg_ckpt_path):
            model.load_seg_checkpoint(seg_ckpt_path)
else:
    print_rank0(f"WARNING: No checkpoint at '{resume_ckpt}'")

# SyncBatchNorm for adapter SPM
model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
model = DDP(model, device_ids=[ddp_info.local_rank], find_unused_parameters=True)

# ---------------------------------------------------------------------------
# Optimizer & Scheduler
# ---------------------------------------------------------------------------

optimizer, optimized_param_dict, all_param_dict = create_seg_source_optimizer(
    model.module, config,
)
optim_param_list = list(optimized_param_dict.values())

total_train_steps = config.training.train_steps
grad_accum_steps = config.training.grad_accum_steps
total_param_update_steps = total_train_steps
total_train_steps = total_train_steps * grad_accum_steps
total_batch_size = batch_size_per_gpu * ddp_info.world_size * grad_accum_steps

scheduler_type = config.training.get("scheduler_type", "cosine")
eta_min = config.training.get("eta_min", 0.0)
lr_scheduler = create_lr_scheduler(
    optimizer,
    total_param_update_steps,
    config.training.warmup,
    scheduler_type=scheduler_type,
    eta_min=eta_min,
)

enable_grad_scaler = config.training.use_amp and config.training.amp_dtype == "fp16"
scaler = torch.amp.GradScaler('cuda', enabled=enable_grad_scaler)
print_rank0(f"Grad scaler enabled: {enable_grad_scaler}")

cur_train_step = 0
cur_param_update_step = 0

# Resume optimizer/scheduler/step from full checkpoint
reset_training = config.training.get("reset_training_state", False)
if _is_full_ckpt and not reset_training:
    if "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
        print_rank0("Restored optimizer state")
    if "lr_scheduler" in ckpt and not config.training.get("reset_lr_scheduler", False):
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        print_rank0("Restored LR scheduler state")
    cur_train_step = ckpt.get("fwdbwd_pass_step", 0)
    cur_param_update_step = ckpt.get("param_update_step", 0)
    print_rank0(f"Resuming from step {cur_train_step} (param_update={cur_param_update_step})")

if _is_full_ckpt:
    del ckpt
    torch.cuda.empty_cache()

dist.barrier()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

start_train_step = cur_train_step
model.train()
_ckpt_save_thread = None

while cur_train_step <= total_train_steps:
    tic = time.time()
    cur_epoch = int(cur_train_step * (total_batch_size / grad_accum_steps) // len(dataset))
    try:
        data = next(dataloader_iter)
    except StopIteration:
        print_rank0(f"Rank {ddp_info.local_rank}: resetting dataloader at epoch {cur_epoch}")
        datasampler.set_epoch(cur_epoch)
        dataloader_iter = iter(dataloader)
        data = next(dataloader_iter)

    # Move tensors to device, keep lists as-is (for ragged labels)
    batch = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(ddp_info.device)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            batch[k] = [t.to(ddp_info.device) for t in v]
        else:
            batch[k] = v

    # Forward
    with torch.autocast(
        enabled=config.training.use_amp,
        device_type="cuda",
        dtype=amp_dtype_mapping[config.training.amp_dtype],
    ):
        ret_dict = model(batch)
        if ret_dict is None:
            continue

    # Backward
    loss = ret_dict.loss
    update_grads = (cur_train_step + 1) % grad_accum_steps == 0 or cur_train_step == total_train_steps

    if not update_grads:
        with model.no_sync():
            scaler.scale(loss / grad_accum_steps).backward()
    else:
        scaler.scale(loss / grad_accum_steps).backward()
    cur_train_step += 1

    export_inter_results = (
        (cur_train_step - 1) == start_train_step
        or cur_train_step % config.training.vis_every == 0
    )

    skip_optimizer_step = False
    if torch.isnan(loss) or torch.isinf(loss):
        print_rank0(f"NaN/Inf loss at step {cur_train_step}, skipping...")
        skip_optimizer_step = True
        loss.data = torch.zeros_like(loss)

    if update_grads and not skip_optimizer_step:
        scaler.unscale_(optimizer)
        with torch.no_grad():
            for p in optimized_param_dict.values():
                if p.requires_grad and p.grad is not None:
                    p.grad.nan_to_num_(nan=0.0, posinf=1e-6, neginf=-1e-6)

        if config.training.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                optim_param_list, max_norm=config.training.grad_clip_norm,
            )

        if not skip_optimizer_step:
            scaler.step(optimizer)
            scaler.update()
            cur_param_update_step += 1

        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # Logging
    if ddp_info.is_main_process:
        log_dict = {}

        # Extract loss components (raw, before weighting)
        nvs_loss = ret_dict.nvs_loss.item() if hasattr(ret_dict, 'nvs_loss') else 0.0
        seg_loss = ret_dict.seg_loss.item() if hasattr(ret_dict, 'seg_loss') else 0.0
        total_loss = ret_dict.loss.item()
        nvs_w = model.module.nvs_loss_weight if hasattr(model, 'module') else model.nvs_loss_weight
        seg_w = model.module.seg_loss_weight if hasattr(model, 'module') else model.seg_loss_weight

        # NVS metrics
        nvs_metrics = {}
        if ret_dict.loss_metrics is not None:
            nvs_metrics = {k: float(f"{v.item():.6f}") for k, v in ret_dict.loss_metrics.items()}

        if (cur_train_step % config.training.print_every == 0) or (cur_train_step < 100 + start_train_step):
            print_str = (
                f"[Epoch {cur_epoch:>3d}] Step {cur_train_step:>6d} "
                f"(Update {cur_param_update_step:>6d}) | "
                f"{time.time() - tic:.2f}s | LR: {optimizer.param_groups[0]['lr']:.2e} | "
                f"Loss: {total_loss:.4f} (nvs: {nvs_loss:.4f}×{nvs_w}, seg: {seg_loss:.4f}×{seg_w})"
            )
            if "psnr" in nvs_metrics:
                print_str += f" | PSNR: {nvs_metrics['psnr']}"
            print(print_str)

        if (cur_train_step % config.training.wandb_log_every == 0) or (cur_train_step < 200 + start_train_step):
            log_dict = {
                "iter": cur_train_step,
                "param_update_step": cur_param_update_step,
                "lr": optimizer.param_groups[0]["lr"],
                "epoch": cur_epoch,
                "train/loss": total_loss,
                "train/nvs_loss": nvs_loss,
                "train/seg_loss": seg_loss,
                "train/weighted_nvs_loss": nvs_w * nvs_loss,
                "train/weighted_seg_loss": seg_w * seg_loss,
            }
            log_dict.update({f"train/{k}": v for k, v in nvs_metrics.items() if k != "loss"})

            # Seg loss breakdown: aggregate ce/mask/dice across aux layers
            if ret_dict.seg_output is not None and ret_dict.seg_output.loss_dict is not None:
                agg = {}
                for k, v in ret_dict.seg_output.loss_dict.items():
                    val = v.item() if isinstance(v, torch.Tensor) else v
                    # keys like "loss_ce_0", "loss_mask_0", ... → group by prefix
                    base = k.rsplit("_", 1)[0] if k[-1].isdigit() else k
                    agg[base] = agg.get(base, 0.0) + val
                for k, v in agg.items():
                    log_dict[f"train/seg_{k}"] = v

        # Qualitative visualization
        if export_inter_results and hasattr(ret_dict, "render") and ret_dict.render is not None:
            try:
                _tgt_img = ret_dict.target.image
                _ren_img = ret_dict.render
                _inp_img = ret_dict.input.image
                n_show = min(_tgt_img.shape[1], 4)
                vis_pairs = []
                for v in range(n_show):
                    gt = _tgt_img[0, v].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    rd = _ren_img[0, v].detach().float().clamp(0, 1).permute(1, 2, 0).cpu().numpy()
                    pair = np.concatenate([gt, rd], axis=1)
                    vis_pairs.append(pair)
                vis_grid = np.concatenate(vis_pairs, axis=0)
                vis_grid = (vis_grid * 255).clip(0, 255).astype(np.uint8)
                log_dict["vis/gt_vs_render"] = wandb.Image(vis_grid, caption="GT (left) | Render (right)")

                n_inp = min(_inp_img.shape[1], 4)
                inp_list = [_inp_img[0, v].float().clamp(0, 1).permute(1, 2, 0).cpu().numpy() for v in range(n_inp)]
                inp_grid = np.concatenate(inp_list, axis=1)
                inp_grid = (inp_grid * 255).clip(0, 255).astype(np.uint8)
                log_dict["vis/input_views"] = wandb.Image(inp_grid, caption="Input views")

                # Seg visualization: source RGB | panoptic overlay
                if ret_dict.seg_output is not None:
                    seg_vis = _seg_vis(
                        ret_dict.seg_output, ret_dict.input.image,
                        model.module.source_seg.processor,
                        image_size=_inp_img.shape[-1],
                    )
                    log_dict["vis/seg_panoptic"] = wandb.Image(seg_vis, caption="Source (left) | Panoptic (right)")
            except Exception as e:
                import traceback
                print(f"[Vis] Error: {e}\n{traceback.format_exc()}")

        if log_dict:
            wandb.log(log_dict, step=cur_train_step)

        # Checkpoint
        if (cur_train_step % config.training.checkpoint_every == 0) or (cur_train_step == total_train_steps):
            if _ckpt_save_thread is not None:
                _ckpt_save_thread.join()

            model_weights = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
            checkpoint = {
                "model": {k: v.cpu() for k, v in model_weights.items()},
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "fwdbwd_pass_step": cur_train_step,
                "param_update_step": cur_param_update_step,
            }
            checkpoint_dir = config.training.checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(checkpoint_dir, f"ckpt_{cur_train_step:016}.pt")

            def _save_ckpt(ckpt, path, step):
                torch.save(ckpt, path)
                print(f"Saved checkpoint at step {step} to {os.path.abspath(path)}")

            _ckpt_save_thread = threading.Thread(
                target=_save_ckpt, args=(checkpoint, ckpt_path, cur_train_step), daemon=True,
            )
            _ckpt_save_thread.start()

        if export_inter_results:
            vis_dir = os.path.join(config.training.checkpoint_dir, f"iter_{cur_train_step:08d}")
            os.makedirs(vis_dir, exist_ok=True)
            visualize_intermediate_results(vis_dir, ret_dict)
            torch.cuda.empty_cache()
            model.train()

    if export_inter_results:
        torch.cuda.empty_cache()
        dist.barrier()

if _ckpt_save_thread is not None:
    _ckpt_save_thread.join()

dist.barrier()
dist.destroy_process_group()
