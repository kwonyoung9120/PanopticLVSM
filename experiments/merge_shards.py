"""Merge shard outputs of eval_panoptic.py into metrics over the full set."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metrics import PQAccumulator


def _pq_from_states(states: list[dict]) -> dict:
    acc = PQAccumulator()
    for name in acc._pq.metric_state:
        total = sum(st[name].to(torch.float64) for st in states)
        setattr(acc._pq, name, total.to(getattr(acc._pq, name).dtype))
    return acc.compute()


def _miou(inter: np.ndarray, union: np.ndarray) -> float:
    ious = [inter[c] / union[c] for c in range(1, len(inter)) if union[c] > 0]
    return float(np.mean(ious)) if ious else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", required=True, help="shard pickles written by eval_panoptic.py --shard-out")
    args = ap.parse_args()

    shards = [pickle.load(open(p, "rb")) for p in args.shards]
    inter = sum(np.array(d["miou_inter"], dtype=np.int64) for d in shards)
    union = sum(np.array(d["miou_union"], dtype=np.int64) for d in shards)
    inter_x = sum(np.array(d["miou_inter_excl"], dtype=np.int64) for d in shards)
    union_x = sum(np.array(d["miou_union_excl"], dtype=np.int64) for d in shards)
    n_pairs = sum(d["n_pairs"] for d in shards)
    psnr = sum(d["psnr_sum"] for d in shards) / max(1, sum(d["psnr_n"] for d in shards))
    ssim = sum(d["ssim_sum"] for d in shards) / max(1, sum(d["ssim_n"] for d in shards))
    total_px = sum(d["total_pixels"] for d in shards)
    uncertain = sum(d["uncertain_pixels"] for d in shards) / max(1, total_px)
    sec = sum(d["total_time"] for d in shards) / max(1, n_pairs)

    pq = _pq_from_states([d["pq_state"] for d in shards])
    pq_x = _pq_from_states([d["pq_state_excl"] for d in shards])

    print(f"[merge] n_pairs={n_pairs} shards={len(shards)}")
    for k, v in [
        ("PQ_novel", pq["PQ"]), ("SQ_novel", pq["SQ"]), ("RQ_novel", pq["RQ"]),
        ("mIoU_novel", _miou(inter, union)), ("PSNR_novel", psnr), ("SSIM_novel", ssim),
        ("seconds_per_scene", sec), ("uncertain_frac", uncertain),
        ("PQ_novel_excl", pq_x["PQ"]), ("SQ_novel_excl", pq_x["SQ"]), ("RQ_novel_excl", pq_x["RQ"]),
        ("mIoU_novel_excl", _miou(inter_x, union_x)),
    ]:
        print(f"METRIC {k}={v:.6f}")


if __name__ == "__main__":
    main()
