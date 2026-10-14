"""Pretrain the source-view query decoder (ViT-Adapter + Mask2Former) on COCO panoptic.

The frozen DINOv2 features match the ScanNet stage; the saved weights initialize
`segmentation.pretrained_seg_ckpt` in configs/train_source_seg.yaml.

    torchrun --nproc_per_node=4 experiments/pretrain_coco_seg.py --config configs/pretrain_coco_seg.yaml
"""

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import add_config_args, load_config
from train.distributed import init_distributed
from train.query_decoder import SourceViewSegmentation
from train.utils import create_lr_scheduler, print_rank0

logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")


class COCOPanoptic(Dataset):
    """COCO panoptic crops as single-view Mask2Former targets (crowd regions are void)."""

    def __init__(self, root: str, split: str, size: int, scale: tuple[float, float]):
        ann = json.load(open(os.path.join(root, "annotations", f"panoptic_{split}.json")))
        self.cat_to_label = {c["id"]: i for i, c in enumerate(sorted(ann["categories"], key=lambda c: c["id"]))}
        self.items = ann["annotations"]
        self.image_dir = os.path.join(root, split)
        self.pan_dir = os.path.join(root, f"panoptic_{split}")
        self.size = size
        self.scale = scale

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        img = Image.open(os.path.join(self.image_dir, item["file_name"].replace(".png", ".jpg"))).convert("RGB")
        pan = Image.open(os.path.join(self.pan_dir, item["file_name"]))

        # Random resize of the short side, random crop, horizontal flip.
        w, h = img.size
        s = self.size * random.uniform(*self.scale) / min(w, h)
        nw, nh = max(self.size, round(w * s)), max(self.size, round(h * s))
        img, pan = img.resize((nw, nh), Image.BILINEAR), pan.resize((nw, nh), Image.NEAREST)
        x, y = random.randint(0, nw - self.size), random.randint(0, nh - self.size)
        box = (x, y, x + self.size, y + self.size)
        img, pan = img.crop(box), pan.crop(box)
        if random.random() < 0.5:
            img, pan = img.transpose(Image.FLIP_LEFT_RIGHT), pan.transpose(Image.FLIP_LEFT_RIGHT)

        rgb = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255.0
        p = np.asarray(pan).astype(np.int64)
        ids = p[..., 0] + 256 * p[..., 1] + 65536 * p[..., 2]
        masks, labels = [], []
        for seg in item["segments_info"]:
            m = ids == seg["id"]
            if seg.get("iscrowd", 0) or not m.any():
                continue
            masks.append(m)
            labels.append(self.cat_to_label[seg["category_id"]])
        if masks:
            masks = torch.from_numpy(np.stack(masks)).float().unsqueeze(1)  # [N, 1, H, W]
        else:
            masks = torch.zeros(0, 1, self.size, self.size)
        return rgb, masks, torch.tensor(labels, dtype=torch.long)


def collate(batch):
    rgb, masks, labels = zip(*batch)
    return torch.stack(rgb).unsqueeze(1), list(masks), list(labels)


def save(model, path, step):
    sd = {f"model.{k}": v for k, v in model.state_dict().items() if not k.startswith("dino_encoder.")}
    torch.save({"state_dict": sd, "step": step}, path)


def main():
    parser = argparse.ArgumentParser()
    add_config_args(parser, "configs/pretrain_coco_seg.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, args.opts)
    t = cfg.training

    ddp = init_distributed(seed=t.get("seed", 42))
    os.makedirs(t.out_dir, exist_ok=True)

    model = SourceViewSegmentation(cfg.segmentation, embed_dim=768, h_patch=16, w_patch=16).to(ddp.device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    params = [p for p in model.parameters() if p.requires_grad]
    print_rank0(f"trainable params: {sum(p.numel() for p in params):,}")
    model = DDP(model, device_ids=[ddp.local_rank], find_unused_parameters=True)

    dataset = COCOPanoptic(cfg.data.root, "train2017", cfg.data.image_size, tuple(cfg.data.scale))
    sampler = DistributedSampler(dataset, shuffle=True, seed=t.get("seed", 42))
    loader = DataLoader(dataset, batch_size=t.batch_size_per_gpu, sampler=sampler, num_workers=t.num_workers,
                        collate_fn=collate, pin_memory=True, drop_last=True, persistent_workers=True)

    optimizer = torch.optim.AdamW(params, lr=t.lr, weight_decay=t.weight_decay, betas=(0.9, 0.999))
    scheduler = create_lr_scheduler(optimizer, t.train_steps, t.warmup, scheduler_type="cosine")

    step, epoch, tic = 0, 0, time.time()
    model.train()
    while step < t.train_steps:
        sampler.set_epoch(epoch)
        for rgb, masks, labels in loader:
            rgb = rgb.to(ddp.device, non_blocking=True)
            masks = [m.to(ddp.device, non_blocking=True) for m in masks]
            labels = [c.to(ddp.device, non_blocking=True) for c in labels]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(rgb, mask_labels=masks, class_labels=labels).seg_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, t.grad_clip_norm)
            if torch.isfinite(grad_norm):
                optimizer.step()
            scheduler.step()
            step += 1

            if step % t.print_every == 0:
                elapsed = time.time() - tic
                eta = elapsed / step * (t.train_steps - step) / 3600
                print_rank0(f"step {step}/{t.train_steps}  loss {loss.item():.4f}  grad {grad_norm.item():.3f}  "
                            f"lr {scheduler.get_last_lr()[0]:.2e}  {elapsed / step:.2f}s/it  eta {eta:.1f}h", flush=True)
            if ddp.is_main_process and (step % t.checkpoint_every == 0 or step == t.train_steps):
                save(model.module, os.path.join(t.out_dir, "model.pt"), step)
            if step >= t.train_steps:
                break
        epoch += 1

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
