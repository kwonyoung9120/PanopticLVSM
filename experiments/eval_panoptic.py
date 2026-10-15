"""Evaluate frozen NVS panoptic propagation on ScanNet validation pairs."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import add_config_args, load_config, print_config, resolve_ckpt
from scannet_loader import ScanNetValPairLoader
from panoptic_codecs import make_codec
from source_seg import make_source_seg
from less3depend_wrapper import Less3Depend
from metrics import PQAccumulator, MIoUAccumulator, pair_concat, psnr, to_uint8_range


def _load_backbone(device: str, ckpt: str = "", ckpt_prefix: str = ""):
    ckpt = resolve_ckpt(ckpt or "ckpts/uplvsm_x224.pt")
    bb = Less3Depend(checkpoint_path=ckpt, ckpt_prefix=ckpt_prefix).to(device).eval()
    return bb, bb.image_size


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(ap, default="configs/scannet_panst3r.yaml")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-out", default="", help="pickle of raw accumulators")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config, args.opts)
    print_config(cfg)

    device = cfg.device
    subset_path = cfg.data.pairs
    codec_args = {k: v for k, v in cfg.codec.items() if k not in ("name", "multipass")}
    codec_name = cfg.codec.name
    shard_idx, num_shards, shard_out = args.shard_idx, args.num_shards, args.shard_out
    n_context = cfg.data.n_context_views
    use_multipass = bool(cfg.codec.get("multipass", False))

    torch.set_grad_enabled(False)

    # Load frozen pipeline components.
    t0 = time.time()
    backbone, nvs_size = _load_backbone(
        device, cfg.nvs.get("ckpt", ""), cfg.nvs.get("ckpt_prefix", "")
    )
    print(f"[load] backbone ready in {time.time()-t0:.1f}s (native size={nvs_size})")

    t0 = time.time()
    codec = make_codec(codec_name, **codec_args)
    size = int(cfg.data.image_size)
    src_seg = make_source_seg(cfg.source_seg, device=device, output_size=size)
    print(f"[load] codec + src_seg ready in {time.time()-t0:.1f}s")

    loader = ScanNetValPairLoader(
        subset_path,
        scannet_root=cfg.data.get("root", ""),
        image_size=cfg.data.image_size,
        max_target_views=cfg.data.max_target_views,
        n_context_views=n_context,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )
    n_pairs = len(loader)
    print(f"[data] {n_pairs} pairs in subset (shard {shard_idx}/{num_shards})")

    # Initialize metric accumulators.
    pq_acc = PQAccumulator()
    miou_acc = MIoUAccumulator()
    # Track a second metric pass with uncertain pixels removed.
    pq_acc_exc = PQAccumulator()
    miou_acc_exc = MIoUAccumulator()
    psnr_vals: List[float] = []
    ssim_vals: List[float] = []
    lpips_vals: List[float] = []
    per_pair_time: List[float] = []
    uncertain_pixels = 0  # total uncertain pixels (across views/pairs)
    total_pixels = 0  # total pixels seen (across views/pairs)
    has_mask_path = codec.name == "binary"
    # SSIM and LPIPS as configured in SIU3R/src/evaluator.py.
    from torchmetrics.image import (
        LearnedPerceptualImagePatchSimilarity,
        StructuralSimilarityIndexMeasure,
    )

    ssim_fn = StructuralSimilarityIndexMeasure(sync_on_compute=False).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(
        "vgg", normalize=True, sync_on_compute=False
    ).to(device)

    # Evaluate each validation pair.
    for idx in range(n_pairs):
        pair = loader[idx]
        t_start = time.time()
        # Source segmentation (either oracle GT or cached panst3r).
        src_out = src_seg(
            source_rgb=pair.context_images,
            source_panoptic=pair.context_panoptic,
            scene=pair.scene,
            context_ids=pair.context_ids,
        )
        src_pano = src_out["pano"].to(device)  # [V_ctx, H, W]

        # If the codec can use per-mask features (VQCodec), stash them.
        if hasattr(codec, "set_scene_features") and "mask_features" in src_out:
            codec.set_scene_features(
                mask_features=src_out["mask_features"],
                mask_ids=src_out["mask_ids"],
                mask_areas=src_out.get("mask_areas"),
            )

        # Encode source panoptic → 3ch signal (at the evaluation resolution).
        # Multi-pass: split instances into chunks of ≤7 across multiple
        # passes, propagate each separately, merge target maps. Address
        # the 3-bit / 8-instance bottleneck without retraining.
        if use_multipass and codec.name == "binary":
            multipass_sigs, multipass_luts = codec.encode_multipass(src_pano)
            src_sig_full, lut = multipass_sigs[0], multipass_luts[0]
        else:
            multipass_sigs, multipass_luts = None, None
            src_sig_full, lut = codec.encode(src_pano)  # [V_ctx, 3, size, size]

        # Prepare NVS inputs at native backbone size.
        ctx_imgs = pair.context_images.unsqueeze(0).to(
            device
        )  # [1, V_ctx, 3, size, size]
        tgt_imgs = pair.target_images.unsqueeze(0).to(device)  # [1, V_tgt, 3, size, size]
        fxfycxcy = pair.fxfycxcy.to(device)
        B, V_ctx = ctx_imgs.shape[:2]
        V_tgt = tgt_imgs.shape[1]

        # Resize RGB to the NVS resolution, adjust intrinsics.
        if nvs_size != size:
            scale = nvs_size / size
            fx, fy, cx, cy = [float(v) * scale for v in pair.fxfycxcy.tolist()]
            fxfycxcy = torch.tensor([fx, fy, cx, cy], device=device)
            ctx_imgs_nvs = F.interpolate(
                ctx_imgs.reshape(B * V_ctx, 3, size, size),
                size=(nvs_size, nvs_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, V_ctx, 3, nvs_size, nvs_size)
            tgt_imgs_nvs = F.interpolate(
                tgt_imgs.reshape(B * V_tgt, 3, size, size),
                size=(nvs_size, nvs_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, V_tgt, 3, nvs_size, nvs_size)
            src_sig_nvs = F.interpolate(
                src_sig_full.reshape(V_ctx, 3, size, size),
                size=(nvs_size, nvs_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(1, V_ctx, 3, nvs_size, nvs_size)
        else:
            ctx_imgs_nvs = ctx_imgs
            tgt_imgs_nvs = tgt_imgs
            src_sig_nvs = src_sig_full.unsqueeze(0)

        # Target pose comes from the target-image hint.
        cond = backbone.estimate(ctx_imgs_nvs, tgt_imgs_nvs, fxfycxcy)
        tgt_rgb_nvs = backbone.render(cond)  # [1, V_tgt, 3, nvs, nvs]
        tgt_sig_nvs = backbone.propagate(src_sig_nvs, cond)

        # Resize outputs back to the evaluation resolution for metric computation.
        def _to_full(x, mode):
            B_, V_, C_, H_, W_ = x.shape
            if (H_, W_) == (size, size):
                return x
            return F.interpolate(
                x.reshape(B_ * V_, C_, H_, W_),
                size=(size, size),
                mode=mode,
                align_corners=False if mode == "bilinear" else None,
            ).reshape(B_, V_, C_, size, size)

        tgt_rgb = _to_full(tgt_rgb_nvs, "bilinear")
        tgt_sig = _to_full(tgt_sig_nvs, "bilinear")

        # Decode target signal → panoptic map.
        if has_mask_path:
            pred_pano, uncertain_mask = codec.decode(
                tgt_sig.squeeze(0),
                lut,
                return_mask=True,
            )  # [V_tgt, size, size], [V_tgt, size, size]
        else:
            pred_pano = codec.decode(tgt_sig.squeeze(0), lut)
            uncertain_mask = None

        # Multi-pass extension: propagate the remaining instance chunks
        # through the same backbone (re-using cond) and merge target
        # maps. Each extra pass costs one extra forward through the
        # propagator but no encoder/estimator re-run.
        if use_multipass and multipass_sigs is not None and len(multipass_sigs) > 1:
            extra_decoded = [pred_pano]
            extra_masks = [uncertain_mask] if has_mask_path else [None]
            for p_extra in range(1, len(multipass_sigs)):
                src_sig_p_full = multipass_sigs[p_extra]
                lut_p = multipass_luts[p_extra]
                if nvs_size != size:
                    src_sig_p_nvs = F.interpolate(
                        src_sig_p_full.reshape(V_ctx, 3, size, size),
                        size=(nvs_size, nvs_size),
                        mode="bilinear",
                        align_corners=False,
                    ).reshape(1, V_ctx, 3, nvs_size, nvs_size)
                else:
                    src_sig_p_nvs = src_sig_p_full.unsqueeze(0)
                tgt_sig_p_nvs = backbone.propagate(src_sig_p_nvs, cond)
                tgt_sig_p = _to_full(tgt_sig_p_nvs, "bilinear")
                if has_mask_path:
                    pred_p, mask_p = codec.decode(
                        tgt_sig_p.squeeze(0), lut_p, return_mask=True
                    )
                    extra_masks.append(mask_p)
                else:
                    pred_p = codec.decode(tgt_sig_p.squeeze(0), lut_p)
                    extra_masks.append(None)
                extra_decoded.append(pred_p)
            pred_pano = codec.decode_merge(extra_decoded, masks=extra_masks)
            if has_mask_path:
                # uncertain pixel = uncertain in ALL passes (intersect)
                u_acc = extra_masks[0]
                for m_extra in extra_masks[1:]:
                    if m_extra is not None:
                        u_acc = u_acc & m_extra
                uncertain_mask = u_acc

        # Accumulate uncertain-pixel stats (after any multi-pass merge).
        if has_mask_path:
            uncertain_pixels += int(uncertain_mask.sum().item())
            total_pixels += int(uncertain_mask.numel())

        # Metrics.
        gt_pano = pair.target_panoptic.to(pred_pano.device)
        gt_rgb = pair.target_images.to(tgt_rgb.device)
        for v in range(V_tgt):
            psnr_vals.append(psnr(tgt_rgb[0, v], gt_rgb[v]))
            pr = to_uint8_range(tgt_rgb[0, v]).unsqueeze(0).to(device)
            gr = to_uint8_range(gt_rgb[v]).unsqueeze(0).to(device)
            ssim_vals.append(float(ssim_fn(pr, gr).item()))
            lpips_vals.append(float(lpips_fn(pr, gr).item()))
        # SIU3R protocol: all target views of a pair are concatenated along
        # height and matched together, once per pair.
        pred_cat, gt_cat = pair_concat(pred_pano), pair_concat(gt_pano)
        pq_acc.update(pred_cat, gt_cat)
        miou_acc.update(pred_cat // 1000, gt_cat // 1000)
        if uncertain_mask is not None:
            u = pair_concat(uncertain_mask)
            pred_exc = torch.where(u, torch.zeros_like(pred_cat), pred_cat)
            gt_exc = torch.where(u, torch.zeros_like(gt_cat), gt_cat)
            pq_acc_exc.update(pred_exc, gt_exc)
            miou_acc_exc.update(pred_exc // 1000, gt_exc // 1000)

        per_pair_time.append(time.time() - t_start)
        if (idx + 1) % 10 == 0:
            avg_s = sum(per_pair_time) / len(per_pair_time)
            print(f"  [{idx+1}/{n_pairs}]  avg {avg_s:.2f} s/pair")

    # Aggregate and emit metrics.
    pq_stats = pq_acc.compute()
    metrics = {
        "PQ_novel": pq_stats["PQ"],
        "SQ_novel": pq_stats["SQ"],
        "RQ_novel": pq_stats["RQ"],
        "mIoU_novel": miou_acc.compute(),
        "PSNR_novel": (sum(psnr_vals) / max(1, len(psnr_vals))),
        "SSIM_novel": (sum(ssim_vals) / max(1, len(ssim_vals))) if ssim_vals else 0.0,
        "LPIPS_novel": (sum(lpips_vals) / max(1, len(lpips_vals))) if lpips_vals else 0.0,
        "seconds_per_scene": sum(per_pair_time) / max(1, len(per_pair_time)),
        "n_pairs": n_pairs,
        "n_context_views": n_context,
    }
    if has_mask_path and total_pixels > 0:
        metrics["uncertain_frac"] = uncertain_pixels / total_pixels
        pq_stats_exc = pq_acc_exc.compute()
        metrics["PQ_novel_excl"] = pq_stats_exc["PQ"]
        metrics["SQ_novel_excl"] = pq_stats_exc["SQ"]
        metrics["RQ_novel_excl"] = pq_stats_exc["RQ"]
        metrics["mIoU_novel_excl"] = miou_acc_exc.compute()
    print(f"[done] {metrics}")

    # Emit the canonical METRIC lines.
    canonical_keys = [
        "PQ_novel",
        "SQ_novel",
        "RQ_novel",
        "mIoU_novel",
        "PSNR_novel",
        "SSIM_novel",
        "LPIPS_novel",
        "seconds_per_scene",
    ]
    if has_mask_path and total_pixels > 0:
        canonical_keys += [
            "uncertain_frac",
            "PQ_novel_excl",
            "SQ_novel_excl",
            "RQ_novel_excl",
            "mIoU_novel_excl",
        ]
    for k in canonical_keys:
        if k in metrics:
            print(f"METRIC {k}={metrics[k]:.6f}")

    # If we're running as a shard, dump the raw accumulator state so a
    # merger can compute global metrics across all shards.
    if shard_out:
        import pickle

        payload = {
            "pq_state": {k: v.detach().cpu() for k, v in pq_acc._pq.metric_state.items()},
            "pq_state_excl": {k: v.detach().cpu() for k, v in pq_acc_exc._pq.metric_state.items()},
            "miou_inter": miou_acc.inter.tolist(),
            "miou_union": miou_acc.union.tolist(),
            "miou_inter_excl": miou_acc_exc.inter.tolist(),
            "miou_union_excl": miou_acc_exc.union.tolist(),
            "psnr_sum": float(sum(psnr_vals)),
            "psnr_n": len(psnr_vals),
            "ssim_sum": float(sum(ssim_vals)),
            "ssim_n": len(ssim_vals),
            "lpips_sum": float(sum(lpips_vals)),
            "lpips_n": len(lpips_vals),
            "uncertain_pixels": int(uncertain_pixels),
            "total_pixels": int(total_pixels),
            "n_pairs": n_pairs,
            "total_time": float(sum(per_pair_time)),
        }
        with open(shard_out, "wb") as f:
            pickle.dump(payload, f)
        print(f"[shard] wrote raw accumulators to {shard_out}")


if __name__ == "__main__":
    main()
