<div align="center">

# Extending a Large View Synthesis Model for Multi-view Panoptic Segmentation

**ECCV 2026**

[Kwonyoung Ryu](https://kwonyoung9120.github.io/)<sup>1,4</sup>, &nbsp;
[In-Jae Lee](https://oliver0922.github.io/)<sup>4</sup>, &nbsp;
[Jonghyun Jin](https://www.linkedin.com/in/jonghyun-jin-096972259/)<sup>2</sup>, &nbsp;
[Hyunjee Lee](https://hyunji12.github.io/)<sup>2</sup>, &nbsp;
[Jongmin Lee](https://bluedream1121.github.io/)<sup>3</sup>, &nbsp;
[Jaesik Park](https://jaesik.info/)<sup>4</sup>

<sup>1</sup> POSTECH &nbsp;&nbsp; <sup>2</sup> POSCO DX &nbsp;&nbsp; <sup>3</sup> Chung-Ang University &nbsp;&nbsp; <sup>4</sup> Seoul National University

[![Project Page](https://img.shields.io/badge/Project-Page-blue?logo=googlechrome&logoColor=white)](https://kwonyoung9120.github.io/PanopticLVSM/)
[![arXiv](https://img.shields.io/badge/arXiv-2607.19765-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.19765)

<img src="docs/teaser.gif" width="88%">

</div>

**TL;DR:** We extend a large view synthesis model into a reconstruction-free panoptic renderer that propagates source-view labels to novel viewpoints.

## Abstract

Large view synthesis models learn accurate cross-view correspondence from RGB supervision alone. We observe that this correspondence generalizes beyond appearance, so binary encoded panoptic labels passed through the model are also propagated to novel views with consistent spatial structure. We present the first work to extend large view synthesis models to 3D scene understanding, reusing a frozen model to propagate panoptic labels to novel views without 3D reconstruction or segmentation-specific training.

## Overview

A frozen large view synthesis model propagates input-view panoptic labels to novel views: labels are encoded as binary channels, passed through the same network that renders RGB, and decoded on the target view. This repo consists of:
1. **Evaluation**: novel-view panoptic segmentation on ScanNet with a plug-in source-view segmenter (PanSt3R or ground truth)
2. **Source-view segmenter training**: the shared query decoder used in the paper's main table
3. **Demo**: a Gradio UI that renders an RGB and a segmentation video between two posed views

## Installation

1. Clone the repo:
```bash
git clone https://github.com/kwonyoung9120/PanopticLVSM.git
cd PanopticLVSM
git submodule update --init --recursive   # nvs_models/, seg_models/
```

2. Set up the environment:
```bash
conda env create -f environment.yml
conda activate panopticlvsm
bash scripts/install_deps.sh
```

## Data & Checkpoints

- ScanNet preprocessed by [SIU3R](https://huggingface.co/datasets/insomnia7/SIU3R/tree/main/scannet), same evaluation protocol. Set `SCANNET_ROOT` to the directory holding `train/`, `val/`, `val_pair.json`.
- Our trained model (fine-tuned NVS + shared query decoder): [`panopticlvsm_scannet.pt`](https://huggingface.co/kwonyoung9120/PanopticLVSM/resolve/main/panopticlvsm_scannet.pt) into `./ckpts`.
- Download the following checkpoints into `./ckpts`: [Less3Depend](https://huggingface.co/littlekoyo/Less3Depend/resolve/main/uplvsm_x224.pt) (`uplvsm_x224.pt`), [PanSt3R](https://download.europe.naverlabs.com/ComputerVision/PanSt3R/panst3r_v2_512_5ds.pth) (`panst3r_v2_512_5ds.pth`, non-commercial), [DINOv2](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_reg4_pretrain.pth) (`dinov2_vitb14_reg4_pretrain.pth`).
- Demo: [LVSM](https://huggingface.co/coast01/LVSM/resolve/main/scene_decoder_only_256.pt) (`scene_decoder_only_256.pt`). The Fig. 5 video script also needs [SAM 2](https://github.com/facebookresearch/sam2#download-checkpoints) (`sam2_hiera_large.pt`).
- Training: COCO panoptic (set `COCO_ROOT` to the directory holding `train2017/`, `panoptic_train2017/`, `annotations/`) and [VGG-19](https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat) (fetched automatically). To skip COCO pretraining, use our [`mask2former_coco_panoptic_dinov2_224.pt`](https://huggingface.co/kwonyoung9120/PanopticLVSM/resolve/main/mask2former_coco_panoptic_dinov2_224.pt) as `segmentation.pretrained_seg_ckpt`.

## Training

Shared query decoder (ViT-Adapter + Mask2Former on frozen DINOv2 features) trained on the input views; the NVS model is trained with photometric losses only.

```bash
# 1. Pretrain the query decoder on COCO panoptic -> runs/pretrain_coco_seg/model.pt
torchrun --nproc_per_node=4 experiments/pretrain_coco_seg.py --config configs/pretrain_coco_seg.yaml

# 2. Joint NVS + panoptic training on ScanNet -> runs/train_source_seg/ckpt_*.pt
torchrun --nproc_per_node=4 experiments/train_source_seg.py --config configs/train_source_seg.yaml \
    segmentation.pretrained_seg_ckpt=runs/pretrain_coco_seg/model.pt
```

## Evaluation

```bash
# Paper protocol (SIU3R accounting): trained model, 2 input views, all target views of val_pair.json, tau 0.2 (DEVICES=0,1,2,3 to pick GPUs)
bash scripts/run_full_val_4gpu.sh configs/scannet_paper.yaml

# Training-free: PanSt3R source segmentation
bash scripts/run_full_val_4gpu.sh configs/scannet_panst3r.yaml

# Single GPU on a subset, with overrides
python experiments/eval_panoptic.py --config configs/scannet_panst3r.yaml data.pairs=splits/subset200.json

# Source-view segmentation only
bash scripts/run_source_val_4gpu.sh

# SAM 2 propagation video (Fig. 5)
python experiments/demo_sam2_video.py --dataset scannet --out_dir demo_out/scannet
```

## Demo

```bash
python app.py                    # http://localhost:7860, PORT=<port> to change
```

A frozen LVSM renders the camera path between two posed views and carries their segmentation along it, producing an RGB video and a segmentation video. Inputs:
- **Two views** (resized to 256x256).
- **Two segmentation images**: one color per segment, the same color for the same segment in both views, black for unlabeled.
- **Cameras**: a JSON file with `c2w` (two 4x4 OpenCV camera-to-world matrices) and `fxfycxcy` (two `[fx, fy, cx, cy]` in pixels at 256x256).

Five ScanNet examples with ground-truth segmentation and cameras are in `docs/examples/` and load from the demo page.

## Acknowledgements

We thank the authors of the following repositories, on which this codebase builds:
- [Less3Depend](https://github.com/ou524u/Less3Depend)
- [SIU3R](https://github.com/WU-CVGL/SIU3R)
- [PanSt3R](https://github.com/naver/panst3r)
- [LVSM](https://github.com/Haian-Jin/LVSM)
- [SAM 2](https://github.com/facebookresearch/sam2)
- [Mask2Former](https://github.com/facebookresearch/Mask2Former), [ViT-Adapter](https://github.com/czczup/ViT-Adapter)

## Citation

```bibtex
@inproceedings{ryu2026extending,
  title     = {Extending a Large View Synthesis Model for Multi-view Panoptic Segmentation},
  author    = {Ryu, Kwonyoung and Lee, In-Jae and Jin, Jonghyun and Lee, Hyunjee and Lee, Jongmin and Park, Jaesik},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```
