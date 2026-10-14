"""Vendored segmentation modules from `seg_models/`."""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SEG_MODELS = Path(os.environ.get("SEG_MODELS_SRC", _REPO_ROOT / "seg_models"))

if str(_SEG_MODELS) not in sys.path:
    sys.path.insert(0, str(_SEG_MODELS))

from mask2former import (  # noqa: E402
    VideoMask2FormerForVideoSegmentation,
    VideoMask2FormerImageProcessor,
)
from vit_adapter import CroCoViTAdapter  # noqa: E402

__all__ = [
    "CroCoViTAdapter",
    "VideoMask2FormerForVideoSegmentation",
    "VideoMask2FormerImageProcessor",
]
