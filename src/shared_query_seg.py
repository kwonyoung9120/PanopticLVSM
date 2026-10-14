"""Trained shared query decoder as the source-view segmenter."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from easydict import EasyDict as edict
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config
from train.query_decoder import SourceViewSegmentation

STUFF_LABELS_1BASED = {1, 2}  # wall, floor


class SharedQuerySourceSeg:
    name = "shared_query"

    def __init__(
        self,
        ckpt_path: str = "ckpts/panopticlvsm_scannet.pt",
        train_config: str = "configs/train_source_seg.yaml",
        output_size: int = 256,
        device: str = "cuda",
    ):
        cfg = edict(OmegaConf.to_container(load_config(train_config), resolve=True))
        patch = cfg.model.image_tokenizer.image_size // cfg.model.image_tokenizer.get("patch_size", 14)
        self.model = SourceViewSegmentation(
            seg_config=cfg.segmentation, embed_dim=cfg.model.transformer.d, h_patch=patch, w_patch=patch
        )
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        sd = {k[len("source_seg."):]: v for k, v in sd.items() if k.startswith("source_seg.")}
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        print(f"[shared_query] loaded {ckpt_path}: {len(sd)} tensors, missing={len(missing)}, unexpected={len(unexpected)}")
        self.model = self.model.to(device).eval()
        self.size = output_size
        self.device = device

    @torch.no_grad()
    def __call__(self, source_rgb: torch.Tensor, source_panoptic=None, scene: str = "", context_ids=None) -> dict:
        out = self.model(source_rgb.unsqueeze(0).to(self.device))
        result = self.model.processor.post_process_panoptic_segmentation(
            outputs=out.seg_output,
            threshold=0.5,
            mask_threshold=0.5,
            overlap_mask_area_threshold=0.8,
            label_ids_to_fuse=STUFF_LABELS_1BASED,
            target_sizes=[(self.size, self.size)],
        )[0]
        seg, infos = result["segmentation"].to(self.device), result["segments_info"]
        pano = torch.zeros_like(seg, dtype=torch.int64)
        counter = {}
        for info in infos:
            sem = info["label_id"] + 1
            counter[sem] = counter.get(sem, 0) + 1
            pano[seg == info["id"]] = sem * 1000 + counter[sem]
        return {"pano": pano, "source": "shared_query"}
