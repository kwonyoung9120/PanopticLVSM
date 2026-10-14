"""Render qualitative Re10K propagation results from cached SAM 2 masks."""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from less3depend_wrapper import Less3Depend
from panoptic_codecs import BinaryCodec

RE10K_ROOT = os.environ.get("RE10K_ROOT", "data/re10k/test")
NVS_SIZE = 224
TARGET_SIZE = 256


def load_re10k_frame(scene: str, frame_idx: int):
    metadata_path = os.path.join(RE10K_ROOT, "metadata", f"{scene}.json")
    with open(metadata_path) as f:
        data = json.load(f)
    frame = data["frames"][frame_idx]
    img = Image.open(frame["image_path"]).convert("RGB")
    fxfycxcy = np.array(frame["fxfycxcy"], dtype=np.float32)
    w2c = np.array(frame["w2c"], dtype=np.float32)
    c2w = np.linalg.inv(w2c)
    # Re10K's fxfycxcy are in PIXEL units of the original image.
    # Center-crop to a square and resize to TARGET_SIZE.
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img_crop = img.crop((left, top, left + side, top + side)).resize(
        (TARGET_SIZE, TARGET_SIZE), Image.LANCZOS
    )
    fxfycxcy_pix = fxfycxcy.copy()
    # Center-crop shifts the principal point.
    fxfycxcy_pix[2] -= left
    fxfycxcy_pix[3] -= top
    # Resize side→TARGET_SIZE scales fx, fy, cx, cy uniformly.
    s = TARGET_SIZE / side
    fxfycxcy_pix *= s
    img_t = torch.from_numpy(np.array(img_crop, dtype=np.float32) / 255.0).permute(
        2, 0, 1
    )
    return img_t, torch.from_numpy(fxfycxcy_pix), torch.from_numpy(c2w)


def colorize_panoptic(pano: torch.Tensor) -> Image.Image:
    """Map sem*1000+inst → RGB image. inst==0 → black."""
    arr = pano.cpu().numpy().astype(np.int64)
    inst = arr % 1000
    palette = np.array(
        [
            [0, 0, 0],  # 0 unlabeled
            [220, 60, 70],
            [60, 180, 75],
            [255, 215, 0],
            [60, 60, 220],
            [200, 100, 220],
            [70, 220, 220],
            [255, 165, 0],
            [180, 180, 180],
        ],
        dtype=np.uint8,
    )
    out = palette[(inst % palette.shape[0])]
    out[arr == 0] = 0
    return Image.fromarray(out)


def compose_figure(
    ctx_imgs, ctx_panos, tgt_img, propagated_pano, propagated_rgb, captions=None
):
    """Make a tiled figure. Rows: per ctx (RGB | mask). Bottom row: target | propagated mask | propagated rgb."""
    # Each tile is 256x256 PIL.
    def to_pil(t):
        if t.dim() == 4:
            t = t[0]
        a = (t.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        if a.shape[0] == 3:
            a = a.transpose(1, 2, 0)
        return Image.fromarray(a)

    # Row 1: ctx0 RGB, ctx0 mask, ctx1 RGB, ctx1 mask
    row1 = []
    for i in range(len(ctx_imgs)):
        row1.append(to_pil(ctx_imgs[i]))
        row1.append(colorize_panoptic(ctx_panos[i]))
    # Row 2: target RGB, propagated mask, propagated rgb
    row2 = [to_pil(tgt_img), colorize_panoptic(propagated_pano), to_pil(propagated_rgb)]
    # Build composite
    tile = TARGET_SIZE
    pad = 6
    n_cols = max(len(row1), len(row2))
    W = n_cols * tile + (n_cols - 1) * pad
    H = 2 * tile + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    for j, im in enumerate(row1):
        canvas.paste(im.resize((tile, tile)), (j * (tile + pad), 0))
    for j, im in enumerate(row2):
        canvas.paste(im.resize((tile, tile)), (j * (tile + pad), tile + pad))
    if captions:
        d = ImageDraw.Draw(canvas)
        d.text((6, 6), captions, fill=(255, 255, 255))
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene_glob", default="cache/re10k_sam2/*_meta.json")
    ap.add_argument("--out_dir", default="cache/re10k_propagate")
    ap.add_argument(
        "--multipass",
        action="store_true",
        help="Split instances into ≤7-codec-passes; merge decoded maps.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda:0"
    torch.set_grad_enabled(False)

    print("[load] Less3Depend...")
    backbone = Less3Depend().to(device).eval()
    codec = BinaryCodec(bits=3, tau=0.2)

    metas = sorted(glob.glob(args.scene_glob))
    print(f"[run] {len(metas)} scenes")
    for meta_path in metas:
        meta = json.load(open(meta_path))
        scene = meta["scene"]
        ctx_ids = meta["context_ids"]
        tgt_ids = meta["target_ids"]
        # Use first target only for figure clarity.
        target_ids = [tgt_ids[len(tgt_ids) // 2]]

        # Load context RGB + SAM2 panos.
        ctx_imgs, ctx_fxfy, ctx_c2w, ctx_panos = [], [], [], []
        cache_dir = os.path.dirname(meta_path)
        for label, vid in [("ctx0", ctx_ids[0]), ("ctx1", ctx_ids[1])]:
            img, fxfy, c2w = load_re10k_frame(scene, vid)
            ctx_imgs.append(img)
            ctx_fxfy.append(fxfy)
            ctx_c2w.append(c2w)
            sam_path = os.path.join(cache_dir, f"{scene}_{label}.npz")
            if not os.path.exists(sam_path):
                print(f"  [skip] missing {sam_path}")
                ctx_panos = None
                break
            npz = np.load(sam_path, allow_pickle=True)
            ctx_panos.append(torch.from_numpy(npz["pano"]))
        if ctx_panos is None or len(ctx_panos) != 2:
            continue

        # Load target RGB
        tgt_imgs, tgt_fxfy, tgt_c2w = [], [], []
        for vid in target_ids:
            img, fxfy, c2w = load_re10k_frame(scene, vid)
            tgt_imgs.append(img)
            tgt_fxfy.append(fxfy)
            tgt_c2w.append(c2w)

        ctx_imgs_t = torch.stack(ctx_imgs).unsqueeze(0)  # [1, 2, 3, 256, 256]
        tgt_imgs_t = torch.stack(tgt_imgs).unsqueeze(0)  # [1, 1, 3, 256, 256]
        # Use shared intrinsics (Re10K's fxfycxcy is roughly constant per scene)
        fxfycxcy = ctx_fxfy[0]
        # Encode source panoptic — multi-pass when >7 instances.
        ctx_panos_stack = torch.stack(ctx_panos)  # [2, 256, 256]
        if args.multipass:
            mp_sigs, mp_luts = codec.encode_multipass(ctx_panos_stack)
        else:
            sig0, lut0 = codec.encode(ctx_panos_stack)
            mp_sigs, mp_luts = [sig0], [lut0]

        # Resize for NVS (compute conditioning once)
        scale = NVS_SIZE / TARGET_SIZE
        fx, fy, cx, cy = (fxfycxcy * scale).tolist()
        fxfycxcy_nvs = torch.tensor([fx, fy, cx, cy], device=device)
        ctx_imgs_nvs = (
            F.interpolate(
                ctx_imgs_t.reshape(2, 3, TARGET_SIZE, TARGET_SIZE),
                size=(NVS_SIZE, NVS_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            .reshape(1, 2, 3, NVS_SIZE, NVS_SIZE)
            .to(device)
        )
        tgt_imgs_nvs = (
            F.interpolate(
                tgt_imgs_t.reshape(len(target_ids), 3, TARGET_SIZE, TARGET_SIZE),
                size=(NVS_SIZE, NVS_SIZE),
                mode="bilinear",
                align_corners=False,
            )
            .reshape(1, len(target_ids), 3, NVS_SIZE, NVS_SIZE)
            .to(device)
        )

        cond = backbone.estimate(ctx_imgs_nvs, tgt_imgs_nvs, fxfycxcy_nvs)
        tgt_rgb_nvs = backbone.render(cond)

        def to_256(x, mode):
            B_, V_, C_, H_, W_ = x.shape
            return F.interpolate(
                x.reshape(B_ * V_, C_, H_, W_),
                size=(TARGET_SIZE, TARGET_SIZE),
                mode=mode,
                align_corners=False if mode == "bilinear" else None,
            ).reshape(B_, V_, C_, TARGET_SIZE, TARGET_SIZE)

        tgt_rgb_256 = to_256(tgt_rgb_nvs, "bilinear")

        # Propagate each codec pass and merge.
        decoded_passes = []
        for sig_p, lut_p in zip(mp_sigs, mp_luts):
            sig_p_nvs = (
                F.interpolate(
                    sig_p.reshape(2, 3, TARGET_SIZE, TARGET_SIZE),
                    size=(NVS_SIZE, NVS_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
                .reshape(1, 2, 3, NVS_SIZE, NVS_SIZE)
                .to(device)
            )
            tgt_sig_nvs = backbone.propagate(sig_p_nvs, cond)
            tgt_sig_256 = to_256(tgt_sig_nvs, "bilinear")
            dec = codec.decode(tgt_sig_256.squeeze(0), lut_p)
            decoded_passes.append(dec)
        pred_pano = codec.decode_merge(decoded_passes)

        # Compose figure.
        fig = compose_figure(
            ctx_imgs=[ctx_imgs[0], ctx_imgs[1]],
            ctx_panos=ctx_panos,
            tgt_img=tgt_imgs[0],
            propagated_pano=pred_pano[0],
            propagated_rgb=tgt_rgb_256[0, 0],
        )
        out_path = os.path.join(args.out_dir, f"{scene}.png")
        fig.save(out_path)
        print(f"  [{scene}] → {out_path}")


if __name__ == "__main__":
    main()
