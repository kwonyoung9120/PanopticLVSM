"""Source-view panoptic segmentation using frozen DINOv2 features."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from transformers import Mask2FormerConfig

from .seg_models import (
    CroCoViTAdapter,
    VideoMask2FormerForVideoSegmentation,
    VideoMask2FormerImageProcessor,
)
from .scannet_classes import PANOPTIC_SEMANTIC2NAME

log = logging.getLogger(__name__)

_CKPTS = os.environ.get("CKPTS_DIR", str(Path(__file__).resolve().parents[2] / "ckpts"))
_DEFAULT_DINO_PATH = os.path.join(_CKPTS, "dinov2_vitb14_reg4_pretrain.pth")


def _build_id2label(num_labels: int = 20) -> dict[int, str]:
    return {i: PANOPTIC_SEMANTIC2NAME.get(i + 1, f"class_{i}") for i in range(num_labels)}


class SourceViewSegmentation(nn.Module):
    """Feed frozen DINOv2 features through ViT-Adapter and Mask2Former."""

    def __init__(self, seg_config, embed_dim: int = 768, h_patch: int = 16, w_patch: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.h_patch = h_patch
        self.w_patch = w_patch

        self.seg_layer_indices = seg_config.get("seg_layer_indices", [2, 5, 8, 11])
        self.interaction_indexes = seg_config.get("interaction_indexes", [2, 5, 8, 11])
        adapter_num_block = seg_config.get("adapter_num_block", 12)

        # DINOv2 input size must be divisible by its 14-pixel patch size.
        self.dino_image_size = seg_config.get("dino_image_size", 224)

        # The adapter input size must be divisible by 32.
        adapter_image_size = seg_config.get("adapter_image_size", h_patch * 16)
        adapter_patchsize = adapter_image_size // h_patch
        self.adapter_image_size = adapter_image_size

        # Frozen feature encoder.
        dino_path = seg_config.get("dino_pretrained_path", _DEFAULT_DINO_PATH)
        self._build_frozen_dino(dino_path)

        # Multi-scale feature adapter.
        self.adapter = CroCoViTAdapter(
            num_block=adapter_num_block,
            embed_dim=embed_dim,
            size=(adapter_image_size, adapter_image_size),
            patchsize=adapter_patchsize,
            interaction_indexes=self.interaction_indexes,
        )

        # Panoptic decoder.
        num_labels = seg_config.get("num_labels", 20)
        num_queries = seg_config.get("num_queries", 100)
        id2label = _build_id2label(num_labels)
        self.mask2former_config = Mask2FormerConfig(
            id2label=id2label,
            num_queries=num_queries,
            train_refer_segmentation=False,
            feature_channels=[embed_dim] * 4,
        )
        self.mask2former = VideoMask2FormerForVideoSegmentation(self.mask2former_config)
        self.processor = VideoMask2FormerImageProcessor()

        # Zero-initialized residual projectors start as identity mappings.
        num_extract = len(self.interaction_indexes)
        self.feat_proj = nn.ModuleList()
        for _ in range(num_extract):
            proj = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, embed_dim),
            )
            nn.init.zeros_(proj[-1].weight)
            nn.init.zeros_(proj[-1].bias)
            self.feat_proj.append(proj)

        # Shared ImageNet normalization for DINOv2 and the adapter.
        self.register_buffer(
            "_imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def _build_frozen_dino(self, pretrained_path: str) -> None:
        """Create and freeze a DINOv2 ViT-B/14 encoder."""
        from .less3depend import vit_base

        self.dino_encoder = vit_base(
            img_size=518,
            patch_size=14,
            num_register_tokens=4,
            interpolate_antialias=True,
            interpolate_offset=0.0,
            block_chunks=0,
            init_values=1.0,
            qk_norm=False,
        )
        state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        missing, unexpected = self.dino_encoder.load_state_dict(state_dict, strict=False)
        log.info(f"[Seg DINOv2] Loaded from {pretrained_path} "
                 f"(missing={len(missing)}, unexpected={len(unexpected)})")
        if hasattr(self.dino_encoder, "mask_token"):
            del self.dino_encoder.mask_token

        for p in self.dino_encoder.parameters():
            p.requires_grad = False
        self.dino_encoder.eval()

    @torch.no_grad()
    def _extract_intermediates(self, images: torch.Tensor) -> list[torch.Tensor]:
        """Return configured DINOv2 layers as `[B*V,N_patches,D]` tensors."""
        # Resize to the encoder's configured resolution.
        if images.shape[-1] != self.dino_image_size:
            images = F.interpolate(
                images, size=(self.dino_image_size, self.dino_image_size),
                mode="bilinear", align_corners=False,
            )
        x = (images - self._imagenet_mean) / self._imagenet_std
        intermediates = self.dino_encoder.get_intermediate_layers(
            x, n=self.seg_layer_indices, reshape=False, norm=False,
        )
        return list(intermediates)

    def _prepare_for_adapter(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample to adapter_image_size and apply ImageNet normalization."""
        if x.shape[-1] != self.adapter_image_size:
            x = F.interpolate(
                x, size=(self.adapter_image_size, self.adapter_image_size),
                mode="bilinear", align_corners=False,
            )
        x = (x - self._imagenet_mean) / self._imagenet_std
        return x

    def _project_intermediates(self, intermediates: list[torch.Tensor]) -> list[torch.Tensor]:
        """Residual projection: feat + proj(feat). Identity at init."""
        return [feat + self.feat_proj[i](feat) for i, feat in enumerate(intermediates)]

    def _build_all_feat(self, projected: list[torch.Tensor]) -> list[torch.Tensor]:
        """Build all_feat list for CroCoViTAdapter at interaction_indexes."""
        BV, N, D = projected[0].shape
        device = projected[0].device
        dtype = projected[0].dtype
        num_adapter_layers = self.adapter.num_block

        all_feat = [torch.zeros(BV, N, D, device=device, dtype=dtype)] * num_adapter_layers
        for feat, idx in zip(projected, self.interaction_indexes):
            all_feat[idx] = feat

        return all_feat

    def forward(
        self,
        source_images: torch.Tensor,
        mask_labels: Optional[list[torch.Tensor]] = None,
        class_labels: Optional[list[torch.Tensor]] = None,
    ) -> edict:
        """Segment `[B,V,3,H,W]` images and return loss, output, and input size."""
        B, V_i = source_images.shape[:2]
        H, W = source_images.shape[3], source_images.shape[4]
        device = source_images.device

        original_size = (H, W)

        # Extract and project frozen DINOv2 features.
        flat_input = source_images.reshape(B * V_i, 3, H, W)
        intermediates = self._extract_intermediates(flat_input)

        projected = self._project_intermediates(intermediates)

        all_feat = self._build_all_feat(projected)

        # Build multi-scale adapter features from normalized RGB.
        spm_input = self._prepare_for_adapter(flat_input)

        multi_scale_feat = self.adapter(spm_input, all_feat)

        # Restore the view dimension for video Mask2Former.
        multi_scale_feat_video = []
        for scale_feat in multi_scale_feat:
            _, d, hi, wi = scale_feat.shape
            multi_scale_feat_video.append(scale_feat.reshape(B, V_i, d, hi, wi))

        seg_output = self.mask2former(
            multi_scale_feat=multi_scale_feat_video,
            mask_labels=mask_labels,
            class_labels=class_labels,
        )

        # Restore prediction masks to the source resolution.
        if seg_output.masks_queries_logits is not None:
            masks = seg_output.masks_queries_logits  # [B, num_queries, V_i, H_m, W_m]
            if masks.shape[-1] != W or masks.shape[-2] != H:
                B_m, num_q, V_m, H_m, W_m = masks.shape
                masks_flat = masks.reshape(B_m * num_q * V_m, 1, H_m, W_m)
                masks_resized = F.interpolate(
                    masks_flat, size=original_size, mode="bilinear", align_corners=False,
                )
                seg_output.masks_queries_logits = masks_resized.reshape(B_m, num_q, V_m, H, W)

        seg_loss = seg_output.loss if seg_output.loss is not None else torch.tensor(0.0, device=device)
        seg_loss = seg_loss / max(V_i, 1)

        return edict(
            seg_loss=seg_loss,
            seg_output=seg_output,
            original_size=original_size,
        )

    def train(self, mode=True):
        """Keep frozen DINOv2 in eval mode."""
        super().train(mode)
        self.dino_encoder.eval()
        return self
