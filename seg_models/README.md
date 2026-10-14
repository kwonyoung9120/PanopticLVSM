# Segmentation models

Third-party segmentation code, kept next to `nvs_models/` (the view synthesis side).

| path | source | used by |
|---|---|---|
| `panst3r/` | [naver/panst3r](https://github.com/naver/panst3r) (submodule) | training-free source-view segmentation |
| `must3r/` | [naver/must3r](https://github.com/naver/must3r) (submodule, pulls DUSt3R + CroCo) | **required** by PanSt3R — its 3D backbone |
| `sam2/` | [facebookresearch/sam2](https://github.com/facebookresearch/sam2) (submodule) | SAM 2 masks for the video demo |
| `mask2former/` | video-adapted [Mask2Former](https://github.com/facebookresearch/Mask2Former), from HuggingFace Transformers | shared query decoder |
| `vit_adapter/` | [ViT-Adapter](https://github.com/czczup/ViT-Adapter) | shared query decoder |

`mask2former/` and `vit_adapter/` are vendored because they carry local
modifications (shared queries across views); the rest are pinned submodules.

MUSt3R is not optional. `PanSt3R.__init__` takes a MUSt3R encoder and decoder,
and `panst3r/panst3r.py` imports `must3r.model` at module level, so PanSt3R
fails to import without it. MUSt3R in turn imports DUSt3R and CroCo, which is
why it carries its own nested submodules — `git submodule update --init
--recursive` brings all three.
