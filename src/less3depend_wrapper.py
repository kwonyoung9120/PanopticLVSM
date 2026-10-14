"""Frozen Less3Depend inference wrapper with reusable scene conditioning."""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import torch
from torch import Tensor, nn
from easydict import EasyDict as edict
from einops import rearrange, repeat

_PKG_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _PKG_ROOT.parent
_CKPTS = Path(os.environ.get("CKPTS_DIR", _REPO_ROOT / "ckpts"))
_LESS3DEPEND_SRC = Path(
    os.environ.get("LESS3DEPEND_SRC", _REPO_ROOT / "nvs_models" / "Less3Depend")
)
_DEFAULT_CKPT = str(_CKPTS / "uplvsm_x224.pt")
_DEFAULT_CONFIG = str(_LESS3DEPEND_SRC / "config" / "uplvsm_x224.yaml")
_DEFAULT_DINOV2 = str(_CKPTS / "dinov2_vitb14_reg4_pretrain.pth")


@contextlib.contextmanager
def _less3depend_env(dinov2_ckpt: str):
    """Set checkpoint paths and stub distributed calls during model setup."""
    import torch.distributed as dist

    os.environ["CKPTS_DIR"] = str(Path(dinov2_ckpt).absolute().parent)

    orig_get_rank = getattr(dist, "get_rank", None)
    orig_barrier = getattr(dist, "barrier", None)
    if orig_get_rank is None or not (
        hasattr(dist, "is_initialized") and dist.is_initialized()
    ):
        dist.get_rank = lambda *a, **k: 0
        dist.barrier = lambda *a, **k: None
    _stub_loss_computer()
    try:
        yield
    finally:
        if orig_get_rank is not None:
            dist.get_rank = orig_get_rank
        if orig_barrier is not None:
            dist.barrier = orig_barrier


def _ensure_less3depend_on_path() -> None:
    p = str(_LESS3DEPEND_SRC)
    if p not in sys.path:
        sys.path.insert(0, p)

def _stub_loss_computer() -> None:
    """Disable inference-only loss construction and its weight download."""
    _ensure_less3depend_on_path()
    import importlib

    loss_mod = importlib.import_module("src.model.loss")
    loss_mod.LossComputer = lambda config: nn.Identity()
    for name in ("src.model.ptlvsm", "src.model.uplvsm"):
        mod = sys.modules.get(name)
        if mod is not None and hasattr(mod, "LossComputer"):
            mod.LossComputer = loss_mod.LossComputer


def _load_config(path: str) -> edict:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path).parameters
    return edict(OmegaConf.to_container(cfg, resolve=True))


class Conditioning:
    """Reusable latent, intrinsics, and estimated target-pose metadata."""

    __slots__ = (
        "latent_rgb",
        "fxfycxcy_src",
        "fxfycxcy_tgt",
        "estimated_pose",
        "image_size",
        "v_src",
        "v_tgt",
        "batch_size",
    )

    def __init__(
        self,
        latent_rgb,
        fxfycxcy_src,
        fxfycxcy_tgt,
        estimated_pose,
        image_size,
        v_src,
        v_tgt,
        batch_size,
    ):
        self.latent_rgb = latent_rgb
        self.fxfycxcy_src = fxfycxcy_src
        self.fxfycxcy_tgt = fxfycxcy_tgt
        self.estimated_pose = estimated_pose
        self.image_size = image_size
        self.v_src = v_src
        self.v_tgt = v_tgt
        self.batch_size = batch_size


class Less3Depend(nn.Module):
    def __init__(
        self,
        checkpoint_path: str = _DEFAULT_CKPT,
        config_path: str = _DEFAULT_CONFIG,
        dinov2_ckpt: str = _DEFAULT_DINOV2,
        image_size: int = 224,
        ckpt_prefix: str = "",
    ):
        super().__init__()
        for p in [checkpoint_path, config_path, dinov2_ckpt]:
            if not Path(p).exists():
                raise FileNotFoundError(p)

        _ensure_less3depend_on_path()
        cfg = _load_config(config_path)
        cfg.inference = getattr(cfg, "inference", edict())
        cfg.inference.if_inference = True
        cfg.training.num_input_views = int(getattr(cfg.training, "num_input_views", 2))
        cfg.training.num_target_views = int(
            getattr(cfg.training, "num_target_views", 1)
        )
        cfg.training.grad_checkpoint_every = 1
        self._cfg = cfg
        self._image_size = image_size

        from src.model.uplvsm import unposedLVSM

        with _less3depend_env(dinov2_ckpt):
            self._model = unposedLVSM(cfg)

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
        if ckpt_prefix:
            sd = {
                k[len(ckpt_prefix) :]: v
                for k, v in sd.items()
                if k.startswith(ckpt_prefix)
            }
            print(f"[Less3Depend] stripped prefix {ckpt_prefix!r}; kept {len(sd)} keys")
        missing, unexpected = self._model.load_state_dict(sd, strict=False)
        if missing:
            print(f"[Less3Depend] {len(missing)} missing keys (first 3: {missing[:3]})")
        if unexpected:
            print(
                f"[Less3Depend] {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})"
            )

        for p in self._model.parameters():
            p.requires_grad = False
        self._model.eval()

        self._tgt_patch_size = int(cfg.model.target_pose_tokenizer.patch_size)

    def train(self, mode: bool = True):
        super().train(mode)
        self._model.eval()
        return self

    # ---------- internals ----------
    def _autocast(self):
        device_type = "cuda" if next(self._model.parameters()).is_cuda else "cpu"
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)

    def _expand_fxfycxcy(self, fxfycxcy: Tensor, b: int, v: int) -> Tensor:
        """Accept (4,), (b,4), or (b,v,4) → return (b,v,4)."""
        f = fxfycxcy
        if f.dim() == 1:
            f = f.view(1, 1, 4).expand(b, v, 4)
        elif f.dim() == 2:
            f = f.unsqueeze(1).expand(b, v, 4)
        else:
            assert f.shape == (b, v, 4), f.shape
        return f.contiguous().to(
            next(self._model.parameters()).device, dtype=torch.float32
        )

    @torch.no_grad()
    def _encode_source(self, source_3ch: Tensor, fxfycxcy_src: Tensor):
        """Encoder pass on any 3-channel source. Returns latent tokens [b, n_latent, d]."""
        m = self._model
        b, v_input, _, h, w = source_3ch.shape
        device = source_3ch.device
        with self._autocast():
            input_img_tokens = m.image_tokenizer(source_3ch)
            _, n_patches, d = input_img_tokens.shape
            input_img_tokens = input_img_tokens.reshape(b, v_input * n_patches, d)

            identity_c2w = (
                torch.eye(4, device=device).expand(b, v_input, 4, 4).contiguous()
            )
            canonical_pose_cond = m.get_pose_from_c2w(
                identity_c2w,
                known_fxfycxcy=fxfycxcy_src,
                image_size=h,
            )
            canonical_pose_tokens = m.canonical_pose_tokenizer(canonical_pose_cond)
            canonical_pose_tokens = canonical_pose_tokens.reshape(
                b, v_input * n_patches, d
            )

            input_img_tokens[:, :n_patches] = (
                input_img_tokens[:, :n_patches] + canonical_pose_tokens[:, :n_patches]
            )

            latent_vector_tokens = m.n_light_field_latent.expand(b, -1, -1)
            enc_input = torch.cat((latent_vector_tokens, input_img_tokens), dim=1)

            intermediate = m.pass_layers(
                m.transformer_encoder,
                enc_input,
                gradient_checkpoint=False,
                checkpoint_every=1,
            )
            n_latent = self._cfg.model.transformer.n_latent_vectors
            latent_tokens, _ = intermediate.split(
                [n_latent, v_input * n_patches],
                dim=1,
            )
        return latent_tokens.float()

    @torch.no_grad()
    def _estimate_target_pose(
        self, target_rgb: Tensor, latent_tokens: Tensor
    ) -> Tensor:
        """Run the estimator on target RGB + scene latent → [b, v_tgt, 7]."""
        m = self._model
        b, v_target = target_rgb.shape[:2]
        with self._autocast():
            repeated_latent = repeat(latent_tokens, "b nl d -> (b v) nl d", v=v_target)
            target_img_tokens = m.image_tokenizer(target_rgb)
            pose_enc = m.estimator(target_img_tokens, repeated_latent)
            pose_enc = rearrange(pose_enc, "(b v) d -> b v d", v=v_target)
        return pose_enc.float()

    @torch.no_grad()
    def _decode(
        self,
        latent_tokens: Tensor,
        estimated_pose_enc: Tensor,
        fxfycxcy_tgt: Tensor,
        v_tgt: int,
        h: int,
        w: int,
    ) -> Tensor:
        m = self._model
        pose_enc_type = getattr(self._cfg, "unposed", edict()).get(
            "pose_enc_type", "absT_quaR"
        )
        with self._autocast():
            target_pose_cond = m.get_pose_from_enc(
                estimated_pose_enc,
                pose_enc_type=pose_enc_type,
                known_fxfycxcy=fxfycxcy_tgt,
                image_size=h,
            )
            target_pose_tokens = m.target_pose_tokenizer(target_pose_cond)
            repeated_latent = repeat(latent_tokens, "b nl d -> (b v) nl d", v=v_tgt)
            decoder_input = torch.cat((target_pose_tokens, repeated_latent), dim=1)
            decoder_input = m.transformer_input_layernorm_decoder(decoder_input)

            tokens_out = m.pass_layers(
                m.transformer_decoder,
                decoder_input,
                gradient_checkpoint=False,
                checkpoint_every=1,
            )
            n_patches = target_pose_tokens.shape[1]
            n_latent = self._cfg.model.transformer.n_latent_vectors
            target_img_tokens, _ = tokens_out.split([n_patches, n_latent], dim=1)
            rendered = m.image_token_decoder(target_img_tokens)
            patch_size = self._tgt_patch_size
            rendered = rearrange(
                rendered,
                "(b v) (hh ww) (p1 p2 c) -> b v c (hh p1) (ww p2)",
                v=v_tgt,
                hh=h // patch_size,
                ww=w // patch_size,
                p1=patch_size,
                p2=patch_size,
                c=3,
            )
        return rendered.float()

    # ---------- public API ----------
    @torch.no_grad()
    def estimate(
        self,
        source_rgb: Tensor,  # [b, v_src, 3, H, W] in [0,1]
        target_rgb: Tensor,  # [b, v_tgt, 3, H, W] in [0,1]  (for pose estimator)
        fxfycxcy: Tensor,  # (4,) | (b,4) | (b,v_src,4) — shared src/tgt
    ) -> Conditioning:
        """Build reusable conditioning via the paper's estimator path:
        encoder on source RGB produces the scene latent; the estimator
        looks at target RGB + latent to predict target pose encodings.
        The latter are reused across the RGB render and arbitrary signal
        propagate calls.
        """
        b, v_src, c, h, w = source_rgb.shape
        v_tgt = target_rgb.shape[1]
        if (h, w) != (self._image_size, self._image_size):
            raise ValueError(
                f"source_rgb {h}x{w} doesn't match backbone image_size "
                f"{self._image_size}; resize before calling."
            )
        device = next(self._model.parameters()).device
        source_rgb = source_rgb.to(device, dtype=torch.float32)
        target_rgb = target_rgb.to(device, dtype=torch.float32)

        fxfycxcy_src = self._expand_fxfycxcy(fxfycxcy, b, v_src)
        fxfycxcy_tgt = self._expand_fxfycxcy(fxfycxcy, b, v_tgt)

        latent_rgb = self._encode_source(source_rgb, fxfycxcy_src)
        estimated_pose = self._estimate_target_pose(target_rgb, latent_rgb)
        return Conditioning(
            latent_rgb=latent_rgb,
            fxfycxcy_src=fxfycxcy_src,
            fxfycxcy_tgt=fxfycxcy_tgt,
            estimated_pose=estimated_pose,
            image_size=(h, w),
            v_src=v_src,
            v_tgt=v_tgt,
            batch_size=b,
        )

    @torch.no_grad()
    @torch.no_grad()
    def render(self, cond: Conditioning) -> Tensor:
        h, w = cond.image_size
        return self._decode(
            cond.latent_rgb,
            cond.estimated_pose,
            cond.fxfycxcy_tgt,
            v_tgt=cond.v_tgt,
            h=h,
            w=w,
        )

    @torch.no_grad()
    def propagate(
        self,
        source_channels: Tensor,  # [b, v_src, 3, H, W]
        cond: Conditioning,
    ) -> Tensor:
        b, v_src, c, h, w = source_channels.shape
        if c != 3:
            raise ValueError(f"source_channels has {c} channels; backbone expects 3")
        if (h, w) != cond.image_size:
            raise ValueError(f"size mismatch: got {(h,w)} expected {cond.image_size}")
        device = cond.latent_rgb.device
        source_channels = source_channels.to(device, dtype=torch.float32)
        latent_sig = self._encode_source(source_channels, cond.fxfycxcy_src)
        return self._decode(
            latent_sig,
            cond.estimated_pose,
            cond.fxfycxcy_tgt,
            v_tgt=cond.v_tgt,
            h=h,
            w=w,
        )

    @property
    def image_size(self) -> int:
        return self._image_size
