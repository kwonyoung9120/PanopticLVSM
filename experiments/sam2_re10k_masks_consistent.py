"""Create cross-view-consistent SAM 2 instance masks for Re10K pairs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile

import numpy as np
import torch
from PIL import Image

_SAM2_SRC = os.environ.get("SAM2_SRC")
if _SAM2_SRC and _SAM2_SRC not in sys.path:
    sys.path.insert(0, _SAM2_SRC)
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

RE10K_ROOT = os.environ.get("RE10K_ROOT", "data/re10k/test")
RE10K_EVAL = os.environ.get("RE10K_EVAL_JSON", "data/re10k/evaluation_index_re10k.json")
SAM2_CKPT = os.environ.get("SAM2_CKPT", "checkpoints/sam2.1_hiera_base_plus.pt")
SAM2_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


def load_re10k_image(scene: str, frame_idx: int, target_size: int = 256) -> np.ndarray:
    metadata_path = os.path.join(RE10K_ROOT, "metadata", f"{scene}.json")
    with open(metadata_path) as f:
        data = json.load(f)
    frame = data["frames"][frame_idx]
    img = Image.open(frame["image_path"]).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side)).resize(
        (target_size, target_size),
        Image.LANCZOS,
    )
    return np.array(img, dtype=np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", required=True)
    ap.add_argument("--out_dir", default="cache/re10k_sam2_consistent")
    ap.add_argument("--target_size", type=int, default=256)
    ap.add_argument(
        "--max_masks",
        type=int,
        default=28,
        help="Cap masks per scene (consistent across both views). "
        "With multi-pass codec, 28 masks → 4 codec passes.",
    )
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[sam2] image-mask generator + video predictor from {SAM2_CKPT}")
    sam2_img = build_sam2(SAM2_CFG, SAM2_CKPT, device="cuda")
    # Match main-paper SAM2 thresholds (sam2_re10k_masks.py v1).
    # These give high-precision masks (each is well-formed) but lower
    # absolute pixel coverage; multi-pass propagation makes that OK.
    mask_gen = SAM2AutomaticMaskGenerator(
        sam2_img,
        points_per_side=24,
        pred_iou_thresh=0.85,
        stability_score_thresh=0.92,
        min_mask_region_area=400,
    )
    video_predictor = build_sam2_video_predictor(SAM2_CFG, SAM2_CKPT, device="cuda")

    eval_idx = json.load(open(RE10K_EVAL))

    for scene in args.scenes:
        if scene not in eval_idx or eval_idx[scene] is None:
            print(f"  [skip] {scene} not in eval index")
            continue
        ctx_ids = eval_idx[scene]["context"]
        tgt_ids = eval_idx[scene]["target"]

        # Skip if already cached.
        out_paths = {
            f"ctx{k}": os.path.join(args.out_dir, f"{scene}_ctx{k}.npz") for k in (0, 1)
        }
        if all(os.path.exists(p) for p in out_paths.values()):
            print(f"  [skip] {scene} already cached")
            continue

        # Load both context views.
        img0 = load_re10k_image(scene, ctx_ids[0], target_size=args.target_size)
        img1 = load_re10k_image(scene, ctx_ids[1], target_size=args.target_size)

        # 1) Seed masks on view 0 via auto mask generator.
        seed_masks = mask_gen.generate(img0)
        # Sort by area, cap to max_masks.
        seed_masks = sorted(seed_masks, key=lambda d: -d["area"])[: args.max_masks]
        if not seed_masks:
            print(f"  [{scene}] no seeds — skipping")
            continue

        # 2) Set up video predictor on a temp dir of two frames.
        with tempfile.TemporaryDirectory() as td:
            Image.fromarray(img0).save(os.path.join(td, "0.jpg"))
            Image.fromarray(img1).save(os.path.join(td, "1.jpg"))
            inference_state = video_predictor.init_state(video_path=td)
            video_predictor.reset_state(inference_state)

            # 3) Inject seed masks into frame 0 with unique obj_ids (1..N).
            obj_id_to_seed = {}
            for k, sd in enumerate(seed_masks):
                seg = sd["segmentation"]  # bool [H, W]
                if seg.shape != (args.target_size, args.target_size):
                    seg = (
                        np.array(
                            Image.fromarray(seg.astype(np.uint8) * 255).resize(
                                (args.target_size, args.target_size), Image.NEAREST
                            )
                        )
                        > 0
                    )
                obj_id = k + 1
                video_predictor.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=obj_id,
                    mask=torch.from_numpy(seg).cuda(),
                )
                obj_id_to_seed[obj_id] = sd

            # 4) Propagate to frame 1.
            frame_to_obj_masks = {0: {}, 1: {}}
            for fr_idx, obj_ids, mask_logits in video_predictor.propagate_in_video(
                inference_state
            ):
                for k, oid in enumerate(obj_ids):
                    m = (mask_logits[k] > 0).squeeze(0).cpu().numpy()  # [H, W]
                    if m.shape != (args.target_size, args.target_size):
                        m = (
                            np.array(
                                Image.fromarray(m.astype(np.uint8) * 255).resize(
                                    (args.target_size, args.target_size), Image.NEAREST
                                )
                            )
                            > 0
                        )
                    frame_to_obj_masks[fr_idx][int(oid)] = m

        # 5) Build pano per view (sem=1, inst=obj_id). Largest instance
        # paints first; smaller wins on overlap (later overwrites).
        for k_view, vid_label in [(0, "ctx0"), (1, "ctx1")]:
            obj_masks = frame_to_obj_masks.get(k_view, {})
            if not obj_masks:
                pano = np.zeros((args.target_size, args.target_size), dtype=np.int64)
            else:
                # Sort obj_ids by area in this view (smaller paints later).
                ordered = sorted(
                    obj_masks.items(),
                    key=lambda kv: -int(kv[1].sum()),
                )
                pano = np.zeros((args.target_size, args.target_size), dtype=np.int64)
                for oid, m in ordered:
                    pano = np.where(m, np.int64(1 * 1000 + oid), pano)
            np.savez(
                out_paths[vid_label],
                pano=pano,
                scene=scene,
                view=ctx_ids[k_view],
                label=vid_label,
                n_masks=len(obj_masks),
            )
            print(
                f"  [{scene}/{vid_label}] {len(obj_masks)} masks → {out_paths[vid_label]}"
            )

        # Metadata for downstream.
        meta_path = os.path.join(args.out_dir, f"{scene}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(
                {
                    "scene": scene,
                    "context_ids": ctx_ids,
                    "target_ids": tgt_ids,
                    "consistent": True,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
