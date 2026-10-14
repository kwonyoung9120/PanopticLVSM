"""Joint NVS and source-view segmentation with separate DINOv2 encoders."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from easydict import EasyDict as edict

from .less3depend import unposedLVSM
from .query_decoder import SourceViewSegmentation

log = logging.getLogger(__name__)


class UnposedLVSMSeg(nn.Module):
    """Combine the unposed LVSM backbone with an independent segmenter."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        seg_cfg = config.segmentation

        # NVS backbone and patch grid.
        self.nvs = unposedLVSM(config)

        image_size = config.model.image_tokenizer.image_size
        patch_size = config.model.image_tokenizer.get("patch_size", 14)
        h_patch = image_size // patch_size
        w_patch = image_size // patch_size

        # Source-view segmentation uses its own frozen DINOv2.
        dino_encoder = self.nvs.image_tokenizer[1]
        embed_dim = getattr(dino_encoder, "embed_dim", None) or dino_encoder.dino_vit.embed_dim
        self.source_seg = SourceViewSegmentation(
            seg_config=seg_cfg,
            embed_dim=embed_dim,
            h_patch=h_patch,
            w_patch=w_patch,
        )

        # Objective weights.
        self.nvs_loss_weight = seg_cfg.get("nvs_loss_weight", 1.0)
        self.seg_loss_weight = seg_cfg.get("seg_loss_weight", 0.05)

    def load_nvs_checkpoint(self, path: str) -> None:
        """Load pretrained NVS (unposedLVSM) weights into self.nvs."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model"] if "model" in ckpt else ckpt

        # Remove the DDP prefix.
        cleaned = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[len("module."):]
            cleaned[k] = v

        # Skip incompatible tensors.
        model_sd = self.nvs.state_dict()
        shape_mismatch = []
        for k in list(cleaned.keys()):
            if k in model_sd and cleaned[k].shape != model_sd[k].shape:
                shape_mismatch.append(k)
                del cleaned[k]

        missing, unexpected = self.nvs.load_state_dict(cleaned, strict=False)
        log.info(f"[NVS] Loaded from {path}")
        log.info(f"  Matched: {len(cleaned) - len(unexpected)}, Missing: {len(missing)}, "
                 f"Unexpected: {len(unexpected)}, Shape mismatch: {len(shape_mismatch)}")
        if missing:
            log.info(f"  Missing (sample): {missing[:5]}")

    def load_seg_checkpoint(self, path: str) -> None:
        """Load COCO-pretrained adapter + mask2former weights into self.source_seg."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

        cleaned = {}
        for key in list(state_dict.keys()):
            clean_key = key[len("model."):] if key.startswith("model.") else key

            if "class_predictor" in clean_key or "criterion" in clean_key or "backbone" in clean_key:
                continue
            elif "queries_embedder" in clean_key or "queries_features" in clean_key:
                pretrained_queries = state_dict[key]
                num_q, q_dim = pretrained_queries.shape
                target_num_q = self.source_seg.mask2former_config.num_queries
                tmp = nn.Embedding(target_num_q, q_dim, device=pretrained_queries.device)
                num_copy = min(num_q, target_num_q)
                tmp.weight.data[:num_copy] = pretrained_queries[:num_copy]
                cleaned[clean_key] = tmp.weight.data
            else:
                cleaned[clean_key] = state_dict[key]

        # Skip incompatible tensors.
        model_sd = self.source_seg.state_dict()
        matched, shape_mismatch = 0, []
        for k in list(cleaned.keys()):
            if k in model_sd:
                if cleaned[k].shape == model_sd[k].shape:
                    matched += 1
                else:
                    shape_mismatch.append(k)
                    del cleaned[k]

        missing, unexpected = self.source_seg.load_state_dict(cleaned, strict=False)
        log.info(f"[Seg] Loaded from {path}")
        log.info(f"  Matched: {matched}, Missing: {len(missing)}, "
                 f"Unexpected: {len(unexpected)}, Shape mismatch: {len(shape_mismatch)}")

    def forward(
        self,
        data_batch: dict,
        has_target_image: bool = True,
        target_has_input: bool = True,
        context_mask_labels: Optional[list[torch.Tensor]] = None,
        context_class_labels: Optional[list[torch.Tensor]] = None,
    ) -> Optional[edict]:
        """Run NVS and optional segmentation losses for an RE10K-format batch."""
        # Read segmentation labels from the batch when omitted.
        if context_mask_labels is None:
            context_mask_labels = data_batch.get("context_mask_labels", None)
        if context_class_labels is None:
            context_class_labels = data_batch.get("context_class_labels", None)

        # Keep ragged segmentation labels out of tensor-only NVS preprocessing.
        _seg_keys = ("context_mask_labels", "context_class_labels",
                     "target_mask_labels", "target_class_labels")
        clean_batch = {k: v for k, v in data_batch.items() if k not in _seg_keys}

        # NVS forward pass.
        nvs_result = self.nvs(
            clean_batch,
            has_target_image=has_target_image,
            target_has_input=target_has_input,
        )
        if nvs_result is None:
            return None

        # Source-view segmentation pass.
        source_images = nvs_result.input.image  # [B, V_i, 3, H, W]
        device = source_images.device

        has_seg_labels = (
            context_mask_labels is not None
            and context_class_labels is not None
            and len(context_mask_labels) > 0
        )

        if has_seg_labels:
            seg_result = self.source_seg(
                source_images=source_images,
                mask_labels=context_mask_labels,
                class_labels=context_class_labels,
            )
            seg_loss = seg_result.seg_loss
            seg_output = seg_result.seg_output
        else:
            seg_loss = torch.tensor(0.0, device=device)
            seg_output = None

        # Combined objective.
        nvs_loss = (
            nvs_result.loss_metrics.loss
            if nvs_result.loss_metrics is not None
            else torch.tensor(0.0, device=device)
        )
        total_loss = self.nvs_loss_weight * nvs_loss + self.seg_loss_weight * seg_loss

        return edict(
            loss=total_loss,
            nvs_loss=nvs_loss,
            loss_metrics=nvs_result.loss_metrics,
            seg_loss=seg_loss,
            seg_output=seg_output,
            render=nvs_result.render,
            input=nvs_result.input,
            target=nvs_result.target,
        )

    def train(self, mode=True):
        super().train(mode)
        self.nvs.loss_computer.eval()
        # source_seg.train() keeps its frozen DINOv2 in eval via override
        return self
