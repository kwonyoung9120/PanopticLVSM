# Render SAM 2/PanSt3R masks and RGB along LVSM camera trajectories.
# Requires LVSM, SAM 2, their checkpoints, and pixelSplat-format Re10K chunks.

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
from PIL import Image

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
LVSM_SRC = Path(os.environ.get("LVSM_SRC", _REPO_ROOT / "nvs_models" / "LVSM"))
CKPTS_DIR = Path(os.environ.get("CKPTS_DIR", _REPO_ROOT / "ckpts"))
sys.path.insert(0, str(LVSM_SRC))
sys.path.insert(0, str(_REPO_ROOT / "src"))

# Optional: a SAM2 source checkout. If unset, `sam2` must be pip-installed.
# Appended (not prepended) so the checkout's own utils/ cannot shadow LVSM's.
SAM2_SRC = os.environ.get("SAM2_SRC", str(_REPO_ROOT / "seg_models" / "sam2"))
if SAM2_SRC and SAM2_SRC not in sys.path:
    sys.path.append(SAM2_SRC)

from panoptic_codecs import BinaryCodec  # noqa: E402

IMG_SIZE = 256

def _make_palette(n: int = 64) -> np.ndarray:
    """index 0 = black (unlabeled); golden-ratio hues for distinct colors."""
    import colorsys

    cols = [[0, 0, 0]]
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        s = 0.9 if i % 2 == 0 else 0.6
        v = 0.95 if i % 3 else 0.7
        cols.append([int(c * 255) for c in colorsys.hsv_to_rgb(h, s, v)])
    return np.array(cols, dtype=np.uint8)


PALETTE = _make_palette()


# pixelSplat input loading.
def build_scene_map(chunks_dir: str):
    """Map scene key -> (chunk_path, example index) by scanning chunk files."""
    scene_map = {}
    for chunk_path in sorted(Path(chunks_dir).glob("*.torch")):
        try:
            examples = torch.load(chunk_path, weights_only=False)
        except Exception as e:
            print(f"  [warn] unreadable chunk {chunk_path.name}: {e}")
            continue
        for i, ex in enumerate(examples):
            scene_map[ex["key"]] = (str(chunk_path), i)
        del examples
    return scene_map


def load_frame(example, frame_idx: int):
    """Decode one frame -> (rgb256 uint8 [H,W,3], fxfycxcy px @256, c2w [4,4]).

    Follows LVSM dataset_scene preprocessing: resize height to 256 (width
    rounded to a patch multiple), then center square crop.
    """
    import io

    cam = example["cameras"][frame_idx].numpy()
    fx, fy, cx, cy = cam[:4]  # normalized by original W/H
    w2c = np.eye(4, dtype=np.float64)
    w2c[:3, :4] = cam[6:].reshape(3, 4)
    c2w = np.linalg.inv(w2c)

    img = Image.open(io.BytesIO(example["images"][frame_idx].numpy().tobytes()))
    W0, H0 = img.size
    fxfycxcy = np.array([fx * W0, fy * H0, cx * W0, cy * H0], dtype=np.float64)

    patch = 8
    resize_h = IMG_SIZE
    resize_w = int(round(resize_h / H0 * W0 / patch) * patch)
    img = img.resize((resize_w, resize_h), resample=Image.LANCZOS)
    fxfycxcy *= (resize_w / W0, resize_h / H0, resize_w / W0, resize_h / H0)

    start_w = (resize_w - IMG_SIZE) // 2
    start_h = (resize_h - IMG_SIZE) // 2
    img = img.crop((start_w, start_h, start_w + IMG_SIZE, start_h + IMG_SIZE))
    fxfycxcy[2] -= start_w
    fxfycxcy[3] -= start_h

    return (
        np.array(img.convert("RGB"), dtype=np.uint8),
        torch.from_numpy(fxfycxcy).float(),
        torch.from_numpy(c2w).float(),
    )


def load_scannet_pair(root: str, entry: dict):
    """ScanNet val pair -> ([rgb256 uint8 x2], [fxfycxcy x2], [c2w x2]).

    ScanNet val images are stored at 256x256 with a matching intrinsic.txt.
    """
    scan = entry["scan"]
    scene_dir = Path(root) / "val" / scan
    intr = np.loadtxt(scene_dir / "intrinsic.txt")
    rgbs, fxfys, c2ws = [], [], []
    for vid in entry["context_ids"]:
        img = Image.open(scene_dir / "color" / f"{vid}.jpg").convert("RGB")
        sx = IMG_SIZE / img.size[0]
        sy = IMG_SIZE / img.size[1]
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        fxfy = torch.tensor(
            [intr[0, 0] * sx, intr[1, 1] * sy, intr[0, 2] * sx, intr[1, 2] * sy]
        ).float()
        c2w = torch.from_numpy(
            np.loadtxt(scene_dir / "extrinsic" / f"{vid}.txt")
        ).float()
        rgbs.append(np.array(img, dtype=np.uint8))
        fxfys.append(fxfy)
        c2ws.append(c2w)
    return rgbs, fxfys, c2ws


def bridge_indices(a: int, b: int, n: int):
    """~n unique frame indices from a to b inclusive (works for a > b too)."""
    ids = np.linspace(a, b, max(n, 2)).round().astype(int).tolist()
    out = []
    for i in ids:
        if i not in out:
            out.append(i)
    return out


def load_scannet_bridge(root: str, entry: dict, n: int):
    """Real intermediate ScanNet frames between the two context ids."""
    scene_dir = Path(root) / "val" / entry["scan"] / "color"
    a, b = entry["context_ids"]
    lo, hi = min(a, b), max(a, b)
    exist = [i for i in range(lo, hi + 1) if (scene_dir / f"{i}.jpg").exists()]
    if a > b:
        exist = exist[::-1]
    if len(exist) < 2 or exist[0] != a or exist[-1] != b:
        exist = [a, b]  # fall back to endpoints only
    picks = sorted(set(np.linspace(0, len(exist) - 1, min(n, len(exist)))
                       .round().astype(int).tolist()))
    rgbs = []
    for k in picks:
        img = Image.open(scene_dir / f"{exist[k]}.jpg").convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        rgbs.append(np.array(img, dtype=np.uint8))
    return rgbs


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
    avg_pose = torch.linalg.inv(avg_pose)
    in_c2ws = avg_pose @ in_c2ws

    scene_scale = scene_scale_factor * torch.max(torch.abs(in_c2ws[:, :3, 3]))
    in_c2ws[:, :3, 3] /= scene_scale
    return in_c2ws


# Cross-view SAM 2 tracking.
class Sam2ConsistentMasker:
    """Cross-view-consistent instance masks via SAM2 video tracking.

    Tracking recipe (vs. the naive 2-frame version):
      1. The SAM2 "video" is the REAL frame sequence bridging ctx0 -> ctx1
         (subsampled), so tracking sees smooth motion instead of one giant
         viewpoint jump.
      2. Seeds come from BOTH endpoints: auto-masks on ctx0 are tracked
         forward; auto-masks on ctx1 that match nothing already tracked
         (IoU < 0.3) get new IDs and are tracked backward.
      3. The final <= max_masks instances are chosen by min(area in ctx0,
         area in ctx1), i.e. objects reliably visible in BOTH views win.
    """

    def __init__(self, ckpt: str, cfg: str, max_masks: int = 7, seed_masks: int = 24):
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        from sam2.build_sam import build_sam2, build_sam2_video_predictor

        sam2_img = build_sam2(cfg, ckpt, device="cuda")
        # Looser than the E5 rebuttal settings (24 / 0.85 / 0.92 / 400):
        # denser proposals so far more instances survive per scene; the
        # multipass codec absorbs the count downstream.
        self.mask_gen = SAM2AutomaticMaskGenerator(
            sam2_img,
            points_per_side=32,
            pred_iou_thresh=0.8,
            stability_score_thresh=0.88,
            min_mask_region_area=200,
        )
        self.video_predictor = build_sam2_video_predictor(cfg, ckpt, device="cuda")
        self.max_masks = max_masks
        self.seed_masks = seed_masks

    @staticmethod
    def _to_256(m: np.ndarray) -> np.ndarray:
        if m.shape != (IMG_SIZE, IMG_SIZE):
            m = np.array(
                Image.fromarray(m.astype(np.uint8) * 255).resize(
                    (IMG_SIZE, IMG_SIZE), Image.NEAREST
                )
            ) > 0
        return m

    def __call__(self, frames: list) -> torch.Tensor:
        """frames: bridge sequence of uint8 RGB [H,W,3]; frames[0]/frames[-1]
        are the two context views. Returns panoptic maps [2, H, W]."""
        last = len(frames) - 1
        seeds0 = self.mask_gen.generate(frames[0])
        seeds0 = sorted(seeds0, key=lambda d: -d["area"])[: self.seed_masks]
        if not seeds0:
            return torch.zeros(2, IMG_SIZE, IMG_SIZE, dtype=torch.int64)

        with tempfile.TemporaryDirectory() as td:
            for i, fr in enumerate(frames):
                Image.fromarray(fr).save(os.path.join(td, f"{i:05d}.jpg"))
            state = self.video_predictor.init_state(video_path=td)
            self.video_predictor.reset_state(state)

            next_oid = 1
            for sd in seeds0:
                self.video_predictor.add_new_mask(
                    inference_state=state, frame_idx=0, obj_id=next_oid,
                    mask=torch.from_numpy(sd["segmentation"]).cuda(),
                )
                next_oid += 1

            # Forward pass: track ctx0 seeds through the bridge to ctx1.
            end_masks = {0: {}, last: {}}
            for fr, obj_ids, logits in self.video_predictor.propagate_in_video(state):
                if fr in end_masks:
                    for k, oid in enumerate(obj_ids):
                        end_masks[fr][int(oid)] = self._to_256(
                            (logits[k] > 0).squeeze(0).cpu().numpy()
                        )

            # Seeds on ctx1 that were NOT reached by forward tracking get new
            # IDs and are tracked backward in a SEPARATE inference state
            # (SAM2 forbids adding object ids once tracking has started).
            tracked_union = np.zeros((IMG_SIZE, IMG_SIZE), dtype=bool)
            for m in end_masks[last].values():
                tracked_union |= m
            seeds1 = sorted(
                self.mask_gen.generate(frames[last]), key=lambda d: -d["area"]
            )[: self.seed_masks]
            new_seeds = []
            for sd in seeds1:
                seg = self._to_256(sd["segmentation"])
                inter = (seg & tracked_union).sum()
                if inter / max(seg.sum(), 1) > 0.3:
                    continue  # already covered by a tracked object
                new_seeds.append(seg)

            if new_seeds:
                state_b = self.video_predictor.init_state(video_path=td)
                self.video_predictor.reset_state(state_b)
                new_oids = []
                for seg in new_seeds:
                    self.video_predictor.add_new_mask(
                        inference_state=state_b, frame_idx=last, obj_id=next_oid,
                        mask=torch.from_numpy(seg).cuda(),
                    )
                    end_masks[last][next_oid] = seg
                    new_oids.append(next_oid)
                    next_oid += 1
                for fr, obj_ids, logits in self.video_predictor.propagate_in_video(
                    state_b, reverse=True
                ):
                    if fr == 0:
                        for k, oid in enumerate(obj_ids):
                            if int(oid) in new_oids:
                                end_masks[0][int(oid)] = self._to_256(
                                    (logits[k] > 0).squeeze(0).cpu().numpy()
                                )

        # Keep the instances most visible in BOTH endpoint views.
        def score(oid):
            a0 = int(end_masks[0].get(oid, np.zeros(1)).sum())
            a1 = int(end_masks[last].get(oid, np.zeros(1)).sum())
            return (min(a0, a1), max(a0, a1))

        all_oids = sorted(
            set(end_masks[0]) | set(end_masks[last]), key=score, reverse=True
        )
        # max_masks <= 0 means keep every tracked instance (multipass codec
        # handles arbitrarily many downstream).
        keep = all_oids if self.max_masks <= 0 else all_oids[: self.max_masks]

        panos = []
        for fr in (0, last):
            pano = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int64)
            visible = [(o, end_masks[fr][o]) for o in keep if o in end_masks[fr]]
            # Largest paints first so smaller instances win on overlap.
            for oid, m in sorted(visible, key=lambda kv: -kv[1].sum()):
                pano = np.where(m, np.int64(1000 + oid), pano)
            panos.append(torch.from_numpy(pano))
        return torch.stack(panos)


# Visualization helpers.
def colorize(pano: torch.Tensor) -> np.ndarray:
    """Color by the FULL id (sem*1000+inst): per-class instance counters
    restart at 1, so instance number alone would collide across classes."""
    ids = pano.cpu().numpy().astype(np.int64)
    idx = 1 + (ids * 2654435761 % 0x7FFFFFFF) % (len(PALETTE) - 1)
    out = PALETTE[idx]
    out[ids == 0] = 0
    return out


def overlay(rgb: np.ndarray, pano: torch.Tensor, alpha: float = 0.45) -> np.ndarray:
    color = colorize(pano)
    mask = (pano.cpu().numpy() > 0)[..., None]
    blend = rgb.astype(np.float32) * (1 - alpha) + color.astype(np.float32) * alpha
    return np.where(mask, blend, rgb.astype(np.float32)).astype(np.uint8)


def make_grid2(columns) -> np.ndarray:
    """Bare 3x2 grid: columns of (RGB top, mask bottom), no captions, no
    spacing — image content only."""
    return np.concatenate(
        [np.concatenate([top, bottom], axis=0) for top, bottom in columns],
        axis=1,
    )


# Demo entry point.
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["re10k", "scannet"], default="re10k")
    ap.add_argument("--chunks_dir", default=str(_REPO_ROOT / "data" / "re10k"),
                    help="pixelSplat-format RealEstate10K test chunks (*.torch)")
    ap.add_argument("--scannet_root",
                    default=os.environ.get("SCANNET_ROOT", str(_REPO_ROOT / "data" / "scannet")))
    ap.add_argument("--scenes", nargs="*", default=None,
                    help="Scene keys; default = all scenes found in both chunks and eval index")
    ap.add_argument("--eval_index",
                    default=str(_REPO_ROOT / "nvs_models/Less3Depend/assets/evaluation_index_re10k.json"))
    ap.add_argument("--config", default=str(LVSM_SRC / "configs/LVSM_scene_decoder_only.yaml"))
    ap.add_argument("--ckpt",
                    default=str(CKPTS_DIR / "scene_decoder_only_256.pt"),
                    help="posed LVSM scene decoder-only 256x256 checkpoint")
    ap.add_argument("--sam2_ckpt", default=str(CKPTS_DIR / "sam2_hiera_large.pt"))
    ap.add_argument("--sam2_cfg", default="configs/sam2/sam2_hiera_l.yaml")
    ap.add_argument("--out_dir", default="demo_out/sam2_video")
    ap.add_argument("--num_frames", type=int, default=40,
                    help="Interpolation steps per pose transition")
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--no_loop", action="store_true", help="Do not loop ctx1 -> ctx0")
    ap.add_argument("--max_masks", type=int, default=0,
                    help="Tracked instances kept; <=0 keeps ALL. Every 7 "
                         "instances adds one multipass LVSM render")
    ap.add_argument("--seed_masks", type=int, default=64,
                    help="SAM2 auto-mask seeds per endpoint before selection")
    ap.add_argument("--masker", choices=["sam2", "panst3r"], default="sam2",
                    help="panst3r = the release's joint multi-view segmenter "
                         "(needs the panst3r submodule / PANST3R_SRC)")
    ap.add_argument("--panst3r_ckpt",
                    default=str(CKPTS_DIR / "panst3r_v2_512_5ds.pth"))
    ap.add_argument("--panst3r_vocab", default="scannetpp100")
    ap.add_argument("--panst3r_include_stuff", action="store_true",
                    help="Also propagate wall/floor (stuff). Default: "
                         "instance (thing) masks only")
    ap.add_argument("--bridge_frames", type=int, default=16,
                    help="Real intermediate frames given to SAM2 for tracking")
    ap.add_argument("--tau", type=float, default=0.1, help="Codec uncertainty threshold")
    ap.add_argument("--max_scenes", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda"
    torch.set_grad_enabled(False)

    if args.dataset == "re10k":
        print(f"[data] scanning chunks in {args.chunks_dir}")
        scene_map = build_scene_map(args.chunks_dir)
        eval_index = json.load(open(args.eval_index))
        jobs = args.scenes or sorted(
            s for s in scene_map if eval_index.get(s) is not None
        )[: args.max_scenes]
        print(f"[data] {len(jobs)} scenes: {jobs}")
    else:  # scannet
        pairs = json.load(open(Path(args.scannet_root) / "val_pair.json"))
        by_scan = {}
        # One pair per scan; prefer moderate view overlap (~0.45 IoU) so the
        # interpolated trajectory actually moves.
        for e in pairs:
            cur = by_scan.get(e["scan"])
            if cur is None or abs(e["iou"] - 0.45) < abs(cur["iou"] - 0.45):
                by_scan[e["scan"]] = e
        jobs = sorted(by_scan.values(), key=lambda e: e["scan"])
        if args.scenes:
            jobs = [e for e in jobs if e["scan"] in args.scenes]
        jobs = jobs[: args.max_scenes]
        print(f"[data] {len(jobs)} scannet pairs: "
              f"{[(e['scan'], e['context_ids']) for e in jobs]}")

    if args.masker == "sam2":
        print("[sam2] loading", args.sam2_ckpt)
        masker = Sam2ConsistentMasker(
            args.sam2_ckpt, args.sam2_cfg, args.max_masks, args.seed_masks
        )
    else:
        print("[panst3r] loading", args.panst3r_ckpt)
        os.environ.setdefault("PANST3R_SRC", str(_REPO_ROOT / "seg_models" / "panst3r" / "src"))
        from source_seg import PanSt3RSourceSeg

        panseg = PanSt3RSourceSeg(
            ckpt_path=args.panst3r_ckpt, vocab=args.panst3r_vocab,
            output_size=IMG_SIZE,
        )

    print("[lvsm] loading", args.ckpt)
    # The demo never computes losses; stub out LossComputer so it does not
    # pull VGG/LPIPS weights or require a torch.distributed group.
    import model.loss as lvsm_loss

    lvsm_loss.LossComputer = lambda config: torch.nn.Identity()
    config = edict(yaml.safe_load(open(args.config)))
    from model.LVSM_scene_decoder_only import Images2LatentScene

    # NOTE: Images2LatentScene.train() returns None, so .eval() cannot be chained.
    model = Images2LatentScene(config).to(device)
    model.eval()
    model.load_ckpt(args.ckpt)

    codec = BinaryCodec(bits=3, tau=args.tau, assign="area")

    import imageio

    chunk_cache = {}
    for job in jobs:
        if args.dataset == "re10k":
            scene = job
            chunk_path, ex_idx = scene_map[scene]
            if chunk_path not in chunk_cache:
                chunk_cache = {chunk_path: torch.load(chunk_path, weights_only=False)}
            example = chunk_cache[chunk_path][ex_idx]
            ctx_ids = eval_index[scene]["context"]
            rgbs, fxfys, c2ws = zip(*[load_frame(example, i) for i in ctx_ids])
            bridge_rgbs = [
                load_frame(example, i)[0]
                for i in bridge_indices(ctx_ids[0], ctx_ids[1], args.bridge_frames)
            ]
        else:
            scene = f"{job['scan']}_{job['context_ids'][0]}_{job['context_ids'][1]}"
            rgbs, fxfys, c2ws = load_scannet_pair(args.scannet_root, job)
            bridge_rgbs = load_scannet_bridge(
                args.scannet_root, job, args.bridge_frames
            )
        if args.masker == "sam2":
            panos = masker(bridge_rgbs)  # [2, H, W] at the two endpoints
        else:
            src = (
                torch.stack([torch.from_numpy(r) for r in rgbs])
                .permute(0, 3, 1, 2).float().div(255.0)
            )
            panos = panseg(src, None, scene, list(range(len(rgbs))))["pano"].cpu()
            if not args.panst3r_include_stuff:
                # Instance (thing) masks only: drop wall/floor, the repo's
                # ScanNet stuff classes (src/metrics.py SCANNET_STUFF_CLASSES).
                sem = panos // 1000
                panos = torch.where(
                    (sem == 1) | (sem == 2), torch.zeros_like(panos), panos
                )
        n_inst = len(torch.unique(panos)) - 1
        if n_inst == 0:
            print(f"  [{scene}] no SAM2 masks, skipping")
            continue

        # Multipass: 7 instances per 3-bit pass, disjoint round-robin subsets.
        mp_sigs, mp_luts = codec.encode_multipass(panos)

        c2w = preprocess_poses(
            torch.stack(c2ws), config.training.get("scene_scale_factor", 1.35)
        ).unsqueeze(0).to(device)                                  # [1, 2, 4, 4]
        fxfycxcy = torch.stack(fxfys).unsqueeze(0).to(device)      # [1, 2, 4]
        imgs = (
            torch.stack([torch.from_numpy(r) for r in rgbs])
            .permute(0, 3, 1, 2).float().div(255.0)
            .unsqueeze(0).to(device)                               # [1, 2, 3, H, W]
        )
        ray_o, ray_d = model.process_data.compute_rays(
            c2w=c2w, fxfycxcy=fxfycxcy, h=IMG_SIZE, w=IMG_SIZE, device=device
        )

        def render(input_image):
            batch = edict(
                input=edict(image=input_image, c2w=c2w, fxfycxcy=fxfycxcy,
                            ray_o=ray_o, ray_d=ray_d),
                target=edict(image_h_w=(IMG_SIZE, IMG_SIZE)),
            )
            with torch.autocast(enabled=True, device_type="cuda", dtype=torch.bfloat16):
                out = model.render_video(
                    batch, traj_type="interpolate", num_frames=args.num_frames,
                    loop_video=not args.no_loop, order_poses=False,
                )
            return out.video_rendering[0].float()  # [F, 3, H, W]

        rgb_frames = render(imgs)
        decoded_passes = []
        for sig_p, lut_p in zip(mp_sigs, mp_luts):
            sig_frames = render(sig_p.unsqueeze(0).to(device))
            decoded_passes.append(codec.decode(sig_frames, lut_p))
        pred_panos = codec.decode_merge(decoded_passes)  # [F, H, W]
        print(f"  [{scene}] {n_inst} instances / {len(mp_sigs)} codec passes, "
              f"{rgb_frames.shape[0]} frames")

        mask0_tile, mask1_tile = colorize(panos[0]), colorize(panos[1])
        frames = []
        for f in range(rgb_frames.shape[0]):
            rgb_f = (rgb_frames[f].clamp(0, 1).numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
            frames.append(make_grid2(
                [(rgbs[0], mask0_tile), (rgbs[1], mask1_tile),
                 (rgb_f, colorize(pred_panos[f]))],
            ))

        out_mp4 = os.path.join(args.out_dir, f"{scene}.mp4")
        imageio.mimsave(out_mp4, frames, fps=args.fps, quality=8)
        mid = len(frames) // 2
        Image.fromarray(frames[mid]).save(os.path.join(args.out_dir, f"{scene}_mid.png"))
        print(f"  [{scene}] -> {out_mp4}")


if __name__ == "__main__":
    main()
