"""Frozen posed LVSM: render RGB and propagate segmentation along a camera path."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from PIL import Image

from config import load_config, resolve_ckpt
from panoptic_codecs import BinaryCodec

_REPO_ROOT = Path(__file__).resolve().parent.parent
SIZE = 256  # LVSM scene decoder-only resolution


def load_image(path_or_img) -> torch.Tensor:
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    img = img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    return torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0


def load_seg(path_or_img) -> np.ndarray:
    """Color-coded segmentation image -> packed 0xRRGGBB, black = unlabeled."""
    img = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    a = np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.NEAREST)).astype(np.int64)
    return (a[..., 0] << 16) | (a[..., 1] << 8) | a[..., 2]


def colors_to_pano(segs: list[np.ndarray]):
    """Same color = same segment across views. Returns [V,H,W] sem*1000+inst ids and id -> RGB."""
    packed = np.stack(segs)
    pano = np.zeros_like(packed)
    id_to_rgb = {}
    for k, c in enumerate(int(c) for c in np.unique(packed) if c != 0):
        iid = (k + 1) * 1000 + 1
        pano[packed == c] = iid
        id_to_rgb[iid] = ((c >> 16) & 255, (c >> 8) & 255, c & 255)
    return torch.from_numpy(pano), id_to_rgb


def colorize(pano: torch.Tensor, id_to_rgb: dict) -> np.ndarray:
    ids = pano.cpu().numpy()
    out = np.zeros((*ids.shape, 3), dtype=np.uint8)
    for iid, rgb in id_to_rgb.items():
        out[ids == iid] = rgb
    return out


def load_cameras(path) -> tuple[torch.Tensor, torch.Tensor]:
    """JSON with per-view `c2w` (4x4, OpenCV) and `fxfycxcy` (pixels at 256x256)."""
    cams = json.load(open(path))
    return torch.tensor(cams["c2w"]).float(), torch.tensor(cams["fxfycxcy"]).float()


def preprocess_poses(in_c2ws: torch.Tensor, scene_scale_factor: float = 1.35):
    """Same normalization as LVSM data.dataset_scene.Dataset.preprocess_poses."""
    center = in_c2ws[:, :3, 3].mean(0)
    avg_forward = F.normalize(in_c2ws[:, :3, 2].mean(0), dim=-1)
    avg_down = in_c2ws[:, :3, 1].mean(0)
    avg_right = F.normalize(torch.cross(avg_down, avg_forward, dim=-1), dim=-1)
    avg_down = F.normalize(torch.cross(avg_forward, avg_right, dim=-1), dim=-1)

    avg_pose = torch.eye(4, device=in_c2ws.device)
    avg_pose[:3, :3] = torch.stack([avg_right, avg_down, avg_forward], dim=-1)
    avg_pose[:3, 3] = center
    in_c2ws = torch.linalg.inv(avg_pose) @ in_c2ws

    scene_scale = scene_scale_factor * torch.max(torch.abs(in_c2ws[:, :3, 3]))
    in_c2ws[:, :3, 3] /= scene_scale
    return in_c2ws


class PanopticLVSM:
    def __init__(self, config: str = "configs/demo.yaml", overrides=None):
        cfg = load_config(config, overrides)
        self.device = cfg.device
        sys.path.insert(0, str(_REPO_ROOT / cfg.nvs.src))
        # Rendering never computes losses; skip VGG/LPIPS and torch.distributed.
        import model.loss as lvsm_loss

        lvsm_loss.LossComputer = lambda config: torch.nn.Identity()
        from model.LVSM_scene_decoder_only import Images2LatentScene

        self.lvsm_cfg = edict(yaml.safe_load(open(_REPO_ROOT / cfg.nvs.src / cfg.nvs.config)))
        self.model = Images2LatentScene(self.lvsm_cfg).to(self.device)
        self.model.eval()  # train() returns None, so this cannot be chained
        self.model.load_ckpt(resolve_ckpt(cfg.nvs.ckpt))
        self.codec = BinaryCodec(**cfg.codec)

    @torch.no_grad()
    def __call__(self, views, segs, cameras, n_frames: int = 60):
        """Two views, their color-coded segmentations, and a camera JSON.

        Renders the path view 1 -> view 2 -> view 1. Returns RGB and segmentation frames
        as uint8 arrays, with the segmentation in the input colors.
        """
        c2w, fxfycxcy = load_cameras(cameras)
        c2w = preprocess_poses(c2w, self.lvsm_cfg.training.get("scene_scale_factor", 1.35))
        c2w = c2w.unsqueeze(0).to(self.device)
        fxfycxcy = fxfycxcy.unsqueeze(0).to(self.device)
        ray_o, ray_d = self.model.process_data.compute_rays(
            c2w=c2w, fxfycxcy=fxfycxcy, h=SIZE, w=SIZE, device=self.device
        )

        def render(image):
            batch = edict(
                input=edict(image=image.unsqueeze(0).to(self.device), c2w=c2w,
                            fxfycxcy=fxfycxcy, ray_o=ray_o, ray_d=ray_d),
                target=edict(image_h_w=(SIZE, SIZE)),
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = self.model.render_video(
                    batch, traj_type="interpolate", num_frames=n_frames, loop_video=True
                )
            return out.video_rendering[0].float()  # [F, 3, H, W]

        rgb = render(torch.stack([load_image(v) for v in views]))

        pano, id_to_rgb = colors_to_pano([load_seg(s) for s in segs])
        # Each 3-bit pass carries 7 segments; more segments add passes.
        sigs, luts = self.codec.encode_multipass(pano.to(self.device))
        pred = self.codec.decode_merge([self.codec.decode(render(s), lut) for s, lut in zip(sigs, luts)])

        to_u8 = lambda x: (x.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        return {
            "rgb": [to_u8(x) for x in rgb],
            "seg": [colorize(p, id_to_rgb) for p in pred],
        }
