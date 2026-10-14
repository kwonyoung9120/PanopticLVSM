"""Less3Depend, imported from its own checkout as `src.model.*`.

Override the locations with `LESS3DEPEND_SRC` and `CKPTS_DIR`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = Path(os.environ.get("LESS3DEPEND_SRC", _REPO_ROOT / "nvs_models" / "Less3Depend"))
os.environ.setdefault("CKPTS_DIR", str(_REPO_ROOT / "ckpts"))

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from src.model.dinov2.vision_transformer import vit_base  # noqa: E402
from src.model.uplvsm import unposedLVSM  # noqa: E402

__all__ = ["unposedLVSM", "vit_base"]
