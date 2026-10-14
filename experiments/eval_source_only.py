"""Evaluate source-view segmentation against ScanNet panoptic labels."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import add_config_args, load_config, print_config
from scannet_loader import ScanNetValPairLoader
from source_seg import make_source_seg
from metrics import PQAccumulator, MIoUAccumulator, pair_concat

SCANNET_NAMES = [
    "unlabeled",
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_config_args(ap, default="configs/scannet_panst3r.yaml")
    ap.add_argument("--shard-idx", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap pairs (0 = all)")
    args = ap.parse_args()
    cfg = load_config(args.config, args.opts)
    print_config(cfg)

    shard_idx, num_shards = args.shard_idx, args.num_shards

    torch.set_grad_enabled(False)
    src_seg = make_source_seg(cfg.source_seg, device=cfg.device, output_size=int(cfg.data.image_size))

    loader = ScanNetValPairLoader(
        cfg.data.pairs,
        scannet_root=cfg.data.get("root", ""),
        image_size=cfg.data.image_size,
        max_target_views=1,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )
    n_pairs = len(loader)
    if args.limit and args.limit < n_pairs:
        n_pairs = args.limit
    print(f"[data] processing {n_pairs} pairs (source-view only)", flush=True)

    pq = PQAccumulator()
    miou = MIoUAccumulator()
    times = []
    batch_size = int(cfg.source_seg.get("batch_size", 8))
    use_batch = hasattr(src_seg, "infer_batch") and batch_size > 1
    print(f"[cfg] batch_size={batch_size}  use_batch={use_batch}", flush=True)

    # SIU3R convention: PQ/mIoU are computed per *pair* with all views of
    # the pair concatenated along H into a single (N*H, W) tensor before
    # the metric update. This matches SIU3R/src/evaluator.py:147-298.
    def _update_pair(src_pano, gt_pano):
        if isinstance(src_pano, torch.Tensor) and src_pano.ndim == 2:
            src_flat = src_pano
        else:
            src_flat = pair_concat(
                src_pano
                if isinstance(src_pano, torch.Tensor)
                else torch.as_tensor(src_pano)
            )
        if isinstance(gt_pano, torch.Tensor) and gt_pano.ndim == 2:
            gt_flat = gt_pano
        else:
            gt_flat = pair_concat(gt_pano)
        pq.update(src_flat, gt_flat)
        miou.update(src_flat // 1000, gt_flat // 1000)

    if use_batch:
        i = 0
        while i < n_pairs:
            bs = min(batch_size, n_pairs - i)
            pairs = [loader[i + k] for k in range(bs)]
            t0 = time.time()
            outs = src_seg.infer_batch(pairs)
            dt = time.time() - t0
            times.append(dt / bs)
            for pair, out in zip(pairs, outs):
                _update_pair(out["pano"], pair.context_panoptic)
            i += bs
            if (i // batch_size) % 5 == 0 or i >= n_pairs:
                print(
                    f"  [{i}/{n_pairs}]  avg {sum(times)/len(times):.2f} s/pair "
                    f"(last batch {dt:.2f}s / {bs})",
                    flush=True,
                )
    else:
        for i in range(n_pairs):
            pair = loader[i]
            t0 = time.time()
            out = src_seg(
                source_rgb=pair.context_images,
                source_panoptic=pair.context_panoptic,
                scene=pair.scene,
                context_ids=pair.context_ids,
            )
            times.append(time.time() - t0)
            _update_pair(out["pano"], pair.context_panoptic)
            if (i + 1) % 25 == 0:
                print(
                    f"  [{i+1}/{n_pairs}]  avg {sum(times)/len(times):.2f} s/pair",
                    flush=True,
                )

    pq_stats = pq.compute()
    m = miou.compute()

    print(
        f"[done] n={n_pairs}  PQ_src={pq_stats['PQ']:.4f}  SQ={pq_stats['SQ']:.4f}  "
        f"RQ={pq_stats['RQ']:.4f}  mIoU_src={m:.4f}  sec/pair={sum(times)/max(1,len(times)):.2f}",
        flush=True,
    )

    print(f"METRIC PQ_src={pq_stats['PQ']:.6f}")
    print(f"METRIC mIoU_src={m:.6f}")
    print(f"METRIC seconds_per_scene={sum(times)/max(1,len(times)):.6f}")

    # Per-class breakdown (torchmetrics only exposes per-class PQ, not
    # TP/FP/FN or SQ/RQ separately, so we show PQ per class + per-class mIoU).
    per_cls_pq = pq_stats.get("per_class", {})
    per_cls_iou = miou.per_class() if hasattr(miou, "per_class") else {}
    print("\nper-class breakdown (sorted by PQ asc):")
    print(f'{"class":>16}  {"PQ":>6}  {"mIoU":>6}')
    rows = []
    for c in sorted(set(per_cls_pq.keys()) | set(per_cls_iou.keys())):
        nm = SCANNET_NAMES[c] if c < len(SCANNET_NAMES) else f"cls-{c}"
        ppq = per_cls_pq.get(c, float("nan"))
        iou = per_cls_iou.get(c, float("nan"))
        rows.append((ppq, nm, iou))
    rows.sort(key=lambda t: (float("inf") if t[0] != t[0] else t[0]))
    for ppq, nm, iou in rows:
        print(f"{nm:>16}  {ppq:>6.3f}  {iou:>6.3f}")


if __name__ == "__main__":
    main()
