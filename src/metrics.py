"""SIU3R-compatible PQ and mIoU metrics for ScanNet panoptic labels.

Follows SIU3R/src/evaluator.py: PQ and mIoU are updated per pair after
concatenating views along height, and both average over all 20 classes
(a class absent from the whole split counts as 0). Class 0 is excluded from
mIoU, while predictions on void pixels remain false positives for the
predicted class. PSNR/SSIM are computed on 8-bit images, as SIU3R scores
the PNGs it saves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set

import torch
import torch.nn.functional as F
from torch import Tensor

# ScanNet constants shared with source_seg.py and SIU3R.
SCANNET_NUM_CLASSES = 21  # 0..20, 0 is "unlabeled"
SCANNET_IGNORE = 0
SCANNET_STUFF_CLASSES: Set[int] = {1, 2}  # wall, floor
SCANNET_THING_CLASSES: Set[int] = {
    c for c in range(1, SCANNET_NUM_CLASSES) if c not in SCANNET_STUFF_CLASSES
}


# RGB reconstruction metrics.
def to_uint8_range(img: Tensor) -> Tensor:
    """Quantize a [0, 1] image the way torchvision's save_image writes PNGs."""
    return (img.float().clamp(0, 1) * 255 + 0.5).floor() / 255


def psnr(pred: Tensor, gt: Tensor) -> float:
    """SIU3R PSNR: 8-bit images, data range = max of the target image.

    SIU3R calls torchmetrics PeakSignalNoiseRatio() without a data range, which
    estimates it per image as max(target) - min(min(target), 0) = max(target).
    """
    pred, gt = to_uint8_range(pred), to_uint8_range(gt)
    mse = F.mse_loss(pred, gt)
    if mse.item() <= 0:
        return float("inf")
    return 10.0 * torch.log10(gt.max() ** 2 / mse).item()


def pair_concat(views: Tensor) -> Tensor:
    """Flatten pair views from `[V, H, W]` to `[V*H, W]`; keep `[H, W]`."""
    if views.ndim == 2:
        return views
    if views.ndim != 3:
        raise ValueError(f"expected [V, H, W] or [H, W], got {tuple(views.shape)}")
    V, H, W = views.shape
    return views.reshape(V * H, W).contiguous()


def miou_from_counts(inter, union, ignore: int = SCANNET_IGNORE) -> float:
    """Mean IoU over every class but `ignore`; a class with no pixels counts as 0 (SIU3R)."""
    ious = [
        float(inter[c]) / float(union[c]) if float(union[c]) > 0 else 0.0
        for c in range(len(inter))
        if c != ignore
    ]
    return float(sum(ious) / max(1, len(ious)))


class MIoUAccumulator:
    """Stream mIoU while excluding class 0 from the final average."""
    def __init__(
        self, num_classes: int = SCANNET_NUM_CLASSES, ignore: int = SCANNET_IGNORE
    ):
        self.K = num_classes
        self.ignore = ignore
        self.inter = torch.zeros(self.K, dtype=torch.int64)
        self.union = torch.zeros(self.K, dtype=torch.int64)

    def update(self, pred: Tensor, target: Tensor) -> None:
        pred = pred.flatten().cpu().to(torch.int64)
        target = target.flatten().cpu().to(torch.int64)
        if pred.shape != target.shape:
            raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
        # Treat out-of-range predictions as void.
        pred = torch.where((pred < 0) | (pred >= self.K), torch.zeros_like(pred), pred)
        K = self.K
        idx = target * K + pred
        hist = torch.bincount(idx, minlength=K * K).reshape(K, K).to(torch.int64)
        tp = torch.diagonal(hist)
        fp = hist.sum(dim=0) - tp
        fn = hist.sum(dim=1) - tp
        self.inter += tp
        self.union += tp + fp + fn

    def compute(self) -> float:
        return miou_from_counts(self.inter, self.union, self.ignore)

    def per_class(self) -> Dict[int, float]:
        """Return {class_index -> IoU} for classes with union > 0 (skip ignore)."""
        out = {}
        for c in range(self.K):
            if c == self.ignore:
                continue
            u = self.union[c].item()
            if u > 0:
                out[c] = self.inter[c].item() / u
        return out


@dataclass
class PQAccumulator:
    """SIU3R-configured PQ for `sem*1000+inst` or native `[..., 2]` labels."""

    def __init__(
        self,
        num_classes: int = SCANNET_NUM_CLASSES,
        things: Set[int] = SCANNET_THING_CLASSES,
        stuffs: Set[int] = SCANNET_STUFF_CLASSES,
    ):
        from torchmetrics.detection import PanopticQuality

        self.K = num_classes
        self.things = sorted(things)
        self.stuffs = sorted(stuffs)
        self._pq = PanopticQuality(
            things=self.things,
            stuffs=self.stuffs,
            return_per_class=True,
            allow_unknown_preds_category=True,
            sync_on_compute=False,
        )

    @staticmethod
    def _encode(pair_label: Tensor) -> Tensor:
        """Split `sem*1000 + inst` into torchmetrics' `[..., 2]` layout."""
        if (
            pair_label.ndim >= 1
            and pair_label.shape[-1] == 2
            and pair_label.dtype == torch.int64
        ):
            return pair_label
        sem = pair_label // 1000
        inst = pair_label % 1000
        return torch.stack([sem, inst], dim=-1)

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update one pair; concatenate multiple views with `pair_concat`."""
        # Match inputs to torchmetrics' state device.
        if pred.is_cuda and not self._pq.iou_sum.is_cuda:
            self._pq = self._pq.to(pred.device)
        pred_enc = self._encode(pred.to(torch.int64))
        target_enc = self._encode(target.to(torch.int64).to(pred.device))
        # Add a batch dimension to a single `[H, W, 2]` label map.
        if pred_enc.ndim == 3:
            pred_enc = pred_enc.unsqueeze(0)
            target_enc = target_enc.unsqueeze(0)
        self._pq.update(pred_enc, target_enc)

    def compute(self) -> Dict[str, float]:
        # Derive SQ and RQ from the same state used by torchmetrics PQ.
        iou_sum = self._pq.iou_sum.detach().cpu().to(torch.float64)
        tp = self._pq.true_positives.detach().cpu().to(torch.float64)
        fp = self._pq.false_positives.detach().cpu().to(torch.float64)
        fn = self._pq.false_negatives.detach().cpu().to(torch.float64)

        denom = tp + 0.5 * fp + 0.5 * fn
        sq = torch.where(tp > 0, iou_sum / tp.clamp(min=1.0), torch.zeros_like(tp))
        rq = torch.where(denom > 0, tp / denom.clamp(min=1.0), torch.zeros_like(tp))
        pq = sq * rq

        # SIU3R averages the per-class values over all classes.
        n_valid = int((denom > 0).sum().item())
        pq_avg = float(pq.mean().item())
        sq_avg = float(sq.mean().item())
        rq_avg = float(rq.mean().item())

        # torchmetrics registers thing classes before stuff classes.
        cat_id_to_idx = self._pq.cat_id_to_continuous_id
        class_order = self.things + self.stuffs

        per_class = {}
        for c in class_order:
            idx = cat_id_to_idx.get(c)
            if idx is None:
                continue
            if denom[idx].item() <= 0:
                continue
            per_class[int(c)] = float(pq[idx].item())

        return {
            "PQ": pq_avg,
            "SQ": sq_avg,
            "RQ": rq_avg,
            "num_classes": n_valid,
            "per_class": per_class,
            "per_class_sq": {
                int(c): float(sq[cat_id_to_idx[c]].item())
                for c in class_order
                if c in cat_id_to_idx and denom[cat_id_to_idx[c]].item() > 0
            },
            "per_class_rq": {
                int(c): float(rq[cat_id_to_idx[c]].item())
                for c in class_order
                if c in cat_id_to_idx and denom[cat_id_to_idx[c]].item() > 0
            },
        }


if __name__ == "__main__":
    # Perfect predictions on 2 of the 20 classes: PQ and mIoU of 2/20 = 0.1,
    # since absent classes count as 0.
    gt = torch.tensor([[1001, 1001, 2005], [1001, 2005, 2005]], dtype=torch.int64)
    pred = gt.clone()
    pq = PQAccumulator()
    pq.update(pred, gt)
    r = pq.compute()

    miou = MIoUAccumulator()
    miou.update(pred // 1000, gt // 1000)
    print(
        f"perfect: PQ={r['PQ']:.4f}  mIoU={miou.compute():.4f}  (per-class PQ={r['per_class']})"
    )

    # Verify pair-wise concatenation.
    gt_mv = torch.stack([gt, gt], dim=0)  # [V=2, H=2, W=3]
    pred_mv = torch.stack([pred, pred], dim=0)
    pq2 = PQAccumulator()
    pq2.update(pair_concat(pred_mv), pair_concat(gt_mv))
    r2 = pq2.compute()

    miou2 = MIoUAccumulator()
    miou2.update(pair_concat(pred_mv) // 1000, pair_concat(gt_mv) // 1000)
    print(f"perfect MV: PQ={r2['PQ']:.4f}  mIoU={miou2.compute():.4f}")

    # Verify predictions on void pixels remain in the confusion matrix.
    gt3 = torch.tensor([[0, 0, 1001]], dtype=torch.int64)
    pred3 = torch.tensor([[3001, 3001, 1001]], dtype=torch.int64)
    miou3 = MIoUAccumulator()
    miou3.update(pred3 // 1000, gt3 // 1000)
    print(f"void-on-pred: mIoU={miou3.compute():.4f}  (expected 0.05: class 1 IoU 1.0, class 3 IoU 0)")
