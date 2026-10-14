#!/bin/bash
# Install runtime dependencies and verify required assets.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"

echo "[install_deps] repo root: $REPO_ROOT"

# Install pip-only dependencies.
python -m pip install --upgrade pip >/dev/null
python -m pip install \
    "transformers>=5.2,<5.6" \
    "wandb>=0.25,<1" \
    "roma>=1.5" \
    "einops>=0.8" \
    "easydict" \
    "omegaconf" \
    "huggingface_hub" \
    "sentencepiece" \
    "trimesh" \
    "matplotlib" \
    "scipy" \
    "rich" \
    "tqdm" \
    "gradio" \
    "imageio" \
    "imageio-ffmpeg" \
    "jaxtyping" \
    "lpips" \
    "scikit-image" \
    "torchmetrics" \
    "plyfile" \
    "xformers==0.0.28.post1" \
    "hydra-core" \
    "iopath"

# Report CUDA availability.
python - <<'PY'
import torch
print(f"[install_deps] torch={torch.__version__}  cuda_available={torch.cuda.is_available()}  gpus={torch.cuda.device_count()}")
PY

# Check the ScanNet root.
SCANNET_ROOT="${SCANNET_ROOT:-$REPO_ROOT/data/scannet}"
if [ -d "$SCANNET_ROOT" ]; then
    echo "[install_deps] SCANNET_ROOT=$SCANNET_ROOT (ok)"
else
    echo "[install_deps] ScanNet root not found at $SCANNET_ROOT"
    echo "   export SCANNET_ROOT=/path/to/scannet"
fi

# Initialize external model submodules when needed.
if [ ! -e "$REPO_ROOT/seg_models/panst3r/src/panst3r/__init__.py" ] \
   || [ ! -e "$REPO_ROOT/seg_models/must3r/dust3r/croco/models" ] \
   || [ ! -e "$REPO_ROOT/nvs_models/Less3Depend/src/model/uplvsm.py" ] \
   || [ ! -e "$REPO_ROOT/nvs_models/LVSM/model" ]; then
    echo "[install_deps] submodules look empty — initialising"
    git -C "$REPO_ROOT" submodule update --init --recursive
fi

for p in "seg_models/panst3r/src/panst3r" "seg_models/must3r/must3r" "seg_models/sam2/sam2" \
         "nvs_models/Less3Depend/src" "nvs_models/LVSM/model"; do
    if [ -e "$REPO_ROOT/$p" ]; then
        echo "  ✓ $p"
    else
        echo "  ✗ $p missing — run: git -C $REPO_ROOT submodule update --init --recursive"
    fi
done

# Apply local compatibility and checkpoint-path patches.
apply_patch() {  # apply_patch <submodule dir> <patch file>
    local dir="$REPO_ROOT/$1" patch="$REPO_ROOT/scripts/patches/$2"
    [ -d "$dir" ] || return 0
    if git -C "$dir" apply --check "$patch" 2>/dev/null; then
        git -C "$dir" apply "$patch" && echo "  ✓ applied $2"
    elif git -C "$dir" apply --reverse --check "$patch" 2>/dev/null; then
        echo "  ✓ $2 already applied"
    else
        echo "  ✗ cannot apply $2 to $1" >&2
        return 1
    fi
}
apply_patch nvs_models/LVSM        lvsm_numpy2_camera_utils.patch
apply_patch nvs_models/Less3Depend less3depend_ckpts_dir.patch
apply_patch seg_models/panst3r     panst3r_lazy_retrieval.patch

# Check required and optional checkpoints.
CKPT_DIR="${CKPTS_DIR:-$REPO_ROOT/ckpts}"
mkdir -p "$CKPT_DIR"
echo "[install_deps] checkpoint dir: $CKPT_DIR"
for f in panopticlvsm_scannet.pt uplvsm_x224.pt panst3r_v2_512_5ds.pth dinov2_vitb14_reg4_pretrain.pth; do
    if [ -f "$CKPT_DIR/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f  — missing. See 'Data & Checkpoint Preparation' in README.md"
    fi
done
for f in scene_decoder_only_256.pt sam2_hiera_large.pt; do
    if [ -f "$CKPT_DIR/$f" ]; then
        echo "  ✓ $f (demo)"
    else
        echo "  - $f  — optional, needed for app.py / experiments/demo_sam2_video.py"
    fi
done
for f in mask2former_coco_panoptic_dinov2_224.pt imagenet-vgg-verydeep-19.mat; do
    if [ -f "$CKPT_DIR/$f" ]; then
        echo "  ✓ $f (training)"
    else
        echo "  - $f  — optional, only needed for experiments/train_source_seg.py"
    fi
done

echo "[install_deps] done. Then:"
echo "  export SCANNET_ROOT=/path/to/scannet"
echo "  bash scripts/run_full_val_4gpu.sh"
