"""Cross-view-consistent SAM 2 instance masks for two input views."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SAM2_SRC = os.environ.get("SAM2_SRC", str(_REPO_ROOT / "seg_models" / "sam2"))
if _SAM2_SRC not in sys.path:
    sys.path.append(_SAM2_SRC)


class SAM2SourceSeg:
    """Track SAM 2 masks across two views and assign panoptic instance IDs."""

    name = "sam2"

    def __init__(
        self,
        ckpt_path: str = "ckpts/sam2_hiera_large.pt",
        model_cfg: str = "configs/sam2/sam2_hiera_l.yaml",
        max_masks: int = 7,
        seed_masks: int = 24,
        output_size: int = 256,
        device: str = "cuda",
    ):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2, build_sam2_video_predictor

        self.mask_gen = SAM2AutomaticMaskGenerator(
            build_sam2(model_cfg, ckpt_path, device=device),
            points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.88,
            min_mask_region_area=200,
        )
        self.video_predictor = build_sam2_video_predictor(model_cfg, ckpt_path, device=device)
        self.max_masks = max_masks
        self.seed_masks = seed_masks
        self.size = output_size
        self.device = device

    def _mask(self, m: np.ndarray) -> np.ndarray:
        if m.shape != (self.size, self.size):
            m = np.array(Image.fromarray(m.astype(np.uint8) * 255).resize((self.size, self.size), Image.NEAREST)) > 0
        return m

    def _track(self, frames, seeds, frame_idx, first_oid, reverse):
        with tempfile.TemporaryDirectory() as td:
            for i, fr in enumerate(frames):
                Image.fromarray(fr).save(os.path.join(td, f"{i:05d}.jpg"))
            state = self.video_predictor.init_state(video_path=td)
            self.video_predictor.reset_state(state)
            oids = []
            for k, seg in enumerate(seeds):
                self.video_predictor.add_new_mask(
                    inference_state=state, frame_idx=frame_idx, obj_id=first_oid + k,
                    mask=torch.from_numpy(seg).to(self.device),
                )
                oids.append(first_oid + k)
            out = {}
            for fr, obj_ids, logits in self.video_predictor.propagate_in_video(state, reverse=reverse):
                out[fr] = {int(o): self._mask((logits[k] > 0).squeeze(0).cpu().numpy()) for k, o in enumerate(obj_ids)}
        return oids, out

    @torch.no_grad()
    def __call__(self, source_rgb: torch.Tensor, source_panoptic=None, scene: str = "", context_ids=None) -> dict:
        frames = [(v.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8) for v in source_rgb]
        last = len(frames) - 1
        n = self.size
        masks = {0: {}, last: {}}

        seeds0 = sorted(self.mask_gen.generate(frames[0]), key=lambda d: -d["area"])[: self.seed_masks]
        if seeds0:
            _, tracked = self._track(frames, [s["segmentation"] for s in seeds0], 0, 1, reverse=False)
            for fr in (0, last):
                masks[fr].update(tracked.get(fr, {}))
        next_oid = 1 + len(seeds0)

        covered = np.zeros((n, n), dtype=bool)
        for m in masks[last].values():
            covered |= m
        seeds1 = []
        for s in sorted(self.mask_gen.generate(frames[last]), key=lambda d: -d["area"])[: self.seed_masks]:
            seg = self._mask(s["segmentation"])
            if (seg & covered).sum() / max(seg.sum(), 1) <= 0.3:
                seeds1.append(seg)
        if seeds1:
            oids, tracked = self._track(frames, seeds1, last, next_oid, reverse=True)
            for oid, seg in zip(oids, seeds1):
                masks[last][oid] = seg
            masks[0].update({o: m for o, m in tracked.get(0, {}).items() if o in oids})

        def score(oid):
            a0 = int(masks[0].get(oid, np.zeros(1)).sum())
            a1 = int(masks[last].get(oid, np.zeros(1)).sum())
            return (min(a0, a1), max(a0, a1))

        keep = sorted(set(masks[0]) | set(masks[last]), key=score, reverse=True)
        if self.max_masks > 0:
            keep = keep[: self.max_masks]

        panos = []
        for fr in (0, last):
            pano = np.zeros((n, n), dtype=np.int64)
            visible = [(o, masks[fr][o]) for o in keep if o in masks[fr]]
            for oid, m in sorted(visible, key=lambda kv: -kv[1].sum()):
                pano = np.where(m, np.int64(1000 + oid), pano)
            panos.append(torch.from_numpy(pano))
        return {"pano": torch.stack(panos), "source": "sam2"}
