"""Load ScanNet validation pairs with RGB, cameras, and panoptic labels."""

from __future__ import annotations

import json
import os
import os.path as osp
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch
from PIL import Image

_SCANNET_ROOT = os.environ.get("SCANNET_ROOT", "data/scannet")


@dataclass
class Pair:
    scene: str
    context_ids: List[int]
    target_ids: List[int]
    context_images: torch.Tensor  # [V_ctx, 3, H, W] in [0,1]
    target_images: torch.Tensor  # [V_tgt, 3, H, W]
    fxfycxcy: torch.Tensor  # [4] (fx, fy, cx, cy) in pixels @ image size
    context_panoptic: torch.Tensor  # [V_ctx, H, W] int64, sem*1000+inst
    target_panoptic: torch.Tensor  # [V_tgt, H, W] int64
    iou: float


def _load_rgb(path: str, size: int) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (3,H,W)


def _load_panoptic(path: str, size: int) -> torch.Tensor:
    """Decode a three-byte ScanNet PNG into `sem*1000+inst` labels."""
    arr = np.asarray(Image.open(path))  # (H,W,3) uint8
    if arr.shape[:2] != (size, size):
        # Preserve discrete labels during resize.
        pil = Image.fromarray(arr).resize((size, size), Image.NEAREST)
        arr = np.asarray(pil)
    label = (
        arr[..., 0].astype(np.int64)
        + arr[..., 1].astype(np.int64) * 256
        + arr[..., 2].astype(np.int64) * 256 * 256
    )
    return torch.from_numpy(label)  # (H,W), int64


def _load_intrinsic(path: str) -> torch.Tensor:
    a = np.loadtxt(path).astype(np.float32)  # may be 4x4 or 3x4
    k = torch.from_numpy(a[:3, :3])
    return k


class ScanNetValPairLoader:
    """Iterate over all or a selected subset of `val_pair.json`."""

    def __init__(
        self,
        subset_path: str,
        image_size: int = 256,
        max_target_views: Optional[int] = 2,
        n_context_views: int = 2,
        scannet_root: str = "",
        val_pair_json: str = "",
        shard_idx: int = 0,
        num_shards: int = 1,
    ):
        scannet_root = scannet_root or _SCANNET_ROOT
        val_pair_json = val_pair_json or osp.join(scannet_root, "val_pair.json")
        with open(val_pair_json, "r") as f:
            self._all_pairs = json.load(f)

        if subset_path == "ALL":
            pairs = list(self._all_pairs)
        else:
            with open(subset_path, "r") as f:
                subset = json.load(f)
            if isinstance(subset, dict) and "indices" in subset:
                pairs = [self._all_pairs[i] for i in subset["indices"]]
            elif isinstance(subset, list):
                pairs = subset
            else:
                raise ValueError(f"Unrecognized subset format in {subset_path}")

        # Interleave pairs deterministically across shards.
        self._pairs = pairs[shard_idx::num_shards] if num_shards > 1 else pairs

        self._size = image_size
        self._max_tgt = max_target_views
        self._n_ctx = int(n_context_views)
        if self._n_ctx < 2:
            raise ValueError(f"n_context_views must be >= 2, got {self._n_ctx}")
        self._root = scannet_root
        self._val_dir = osp.join(self._root, "val")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Pair:
        pair = self._pairs[idx]
        scan_path = osp.join(self._val_dir, pair["scan"])
        ctx_ids = list(pair["context_ids"])
        tgt_ids = list(pair["target_ids"])
        # Promote trailing targets while preserving the leading evaluation targets.
        if self._n_ctx > len(ctx_ids):
            n_promote = self._n_ctx - len(ctx_ids)
            tgt_filtered = [t for t in tgt_ids if t not in ctx_ids]
            promoted = tgt_filtered[-n_promote:] if n_promote > 0 else []
            ctx_ids = ctx_ids + promoted
            tgt_ids = [t for t in tgt_ids if t not in ctx_ids]
        if self._max_tgt is not None and len(tgt_ids) > self._max_tgt:
            # Remove context duplicates before truncating targets.
            tgt_ids = [t for t in tgt_ids if t not in ctx_ids][: self._max_tgt]

        size = self._size

        def _rgb(vid):
            return _load_rgb(osp.join(scan_path, "color", f"{vid}.jpg"), size)

        def _pan(vid):
            return _load_panoptic(osp.join(scan_path, "panoptic", f"{vid}.png"), size)

        ctx_imgs = torch.stack([_rgb(v) for v in ctx_ids], dim=0)
        tgt_imgs = torch.stack([_rgb(v) for v in tgt_ids], dim=0)
        ctx_pan = torch.stack([_pan(v) for v in ctx_ids], dim=0)
        tgt_pan = torch.stack([_pan(v) for v in tgt_ids], dim=0)

        k = _load_intrinsic(osp.join(scan_path, "intrinsic.txt"))
        # Scale the native 256-pixel intrinsics when resizing.
        if size != 256:
            s = float(size) / 256.0
            k = k.clone()
            k[0, 0] *= s
            k[0, 2] *= s
            k[1, 1] *= s
            k[1, 2] *= s
        fxfycxcy = torch.tensor(
            [float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])],
            dtype=torch.float32,
        )

        return Pair(
            scene=pair["scan"],
            context_ids=ctx_ids,
            target_ids=tgt_ids,
            context_images=ctx_imgs,
            target_images=tgt_imgs,
            fxfycxcy=fxfycxcy,
            context_panoptic=ctx_pan,
            target_panoptic=tgt_pan,
            iou=float(pair.get("iou", -1.0)),
        )


def build_default_subset(stride: int = 37, n: int = 50) -> List[int]:
    """Select up to `n` unique scenes by deterministic striding."""
    with open(osp.join(_SCANNET_ROOT, "val_pair.json"), "r") as f:
        pairs = json.load(f)
    total = len(pairs)
    # Keep the first strided pair for each scene.
    seen = set()
    idxs = []
    i = 0
    while len(idxs) < n and i < total:
        cand = (i * stride) % total
        scene = pairs[cand]["scan"]
        if scene in seen:
            i += 1
            continue
        seen.add(scene)
        idxs.append(cand)
        i += 1
    return idxs


if __name__ == "__main__":
    # Print one sample for a local smoke test.
    import tempfile

    subset_idxs = build_default_subset(n=3)
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"indices": subset_idxs}, f)
        path = f.name
    loader = ScanNetValPairLoader(path)
    p = loader[0]
    print(f"scene={p.scene}  ctx_ids={p.context_ids}  tgt_ids={p.target_ids}")
    print(
        f"  context_images: {tuple(p.context_images.shape)}  range=[{p.context_images.min():.3f},{p.context_images.max():.3f}]"
    )
    print(
        f"  context_panoptic: {tuple(p.context_panoptic.shape)}  dtype={p.context_panoptic.dtype}"
    )
    u = torch.unique(p.context_panoptic[0])
    print(f"  unique panoptic IDs view0: {u.numel()}  (first 10: {u[:10].tolist()})")
    print(f"  fxfycxcy={p.fxfycxcy.tolist()}")
