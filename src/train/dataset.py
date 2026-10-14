"""ScanNet dataset in RE10K format with optional panoptic supervision."""

from __future__ import annotations

import json
import logging
import os
import os.path as osp
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset as _Dataset

log = logging.getLogger(__name__)


class Dataset(_Dataset):
    """ScanNet scenes in RE10K format for the original UPLVSM train.py."""

    def __init__(self, config, mode: str = "train"):
        super().__init__()
        self.config = config
        self.mode = mode

        scannet_root = config.training.get("scannet_root", "./datasets/scannet")
        self.image_size = config.model.image_tokenizer.image_size
        self.num_views = config.training.num_views  # total views per sample
        self.num_input_views = config.training.num_input_views
        self.num_target_views = config.training.num_target_views
        self.target_has_input = config.training.get("target_has_input", False)
        self.scene_scale_factor = config.training.get("scene_scale_factor", 1.35)

        # Optional panoptic supervision.
        seg_cfg = config.get("segmentation", {})
        self.load_panoptic = seg_cfg.get("load_panoptic", False)
        self.load_target_panoptic = seg_cfg.get("load_target_panoptic", False)
        if self.load_panoptic or self.load_target_panoptic:
            from .seg_models import VideoMask2FormerImageProcessor
            self.processor = VideoMask2FormerImageProcessor(
                size=(self.image_size, self.image_size),
                reduce_labels=True,
                do_rescale=False,
                do_normalize=False,
                ignore_index=255,
                num_labels=seg_cfg.get("num_labels", 20),
            )

        # SIU3R-compatible view selection thresholds.
        vs_cfg = config.training.get("view_selector", {})
        self.min_frame_dist = vs_cfg.get("min_frame_dist", 10)
        self.max_frame_dist = vs_cfg.get("max_frame_dist", 100)
        self.iou_min = vs_cfg.get("iou_min", 0.3)
        self.iou_max = vs_cfg.get("iou_max", 0.8)

        self.val_pairs = None

        if mode == "train":
            self.scans_dir = osp.join(scannet_root, "train")
            # Index valid scans and overlap matrices.
            self.scans: list[tuple[str, list[int], torch.Tensor]] = []
            for scan_name in sorted(os.listdir(self.scans_dir)):
                scan_path = osp.join(self.scans_dir, scan_name)
                if not osp.isdir(scan_path):
                    continue
                depth_dir = osp.join(scan_path, "depth")
                iou_path = osp.join(scan_path, "iou.pt")
                if not osp.exists(depth_dir) or not osp.exists(iou_path):
                    continue
                frame_ids = sorted(
                    int(f.split(".")[0])
                    for f in os.listdir(depth_dir)
                    if f.endswith(".png")
                )
                if len(frame_ids) >= self.num_views:
                    iou = torch.load(iou_path, weights_only=True)
                    self.scans.append((scan_name, frame_ids, iou))

            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"[ScanNetRE10K] train: {len(self.scans)} scans from {self.scans_dir}")
        else:
            # Use fixed validation pairs during evaluation.
            val_pair_json = config.training.get("val_pair_json", "val_pair.json")
            val_pair_path = osp.join(scannet_root, val_pair_json)
            with open(val_pair_path) as f:
                self.val_pairs = json.load(f)
            self.scans_dir = osp.join(scannet_root, "val")
            if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
                print(f"[ScanNetRE10K] eval: {len(self.val_pairs)} pairs from {val_pair_path}")

    def __len__(self) -> int:
        if self.val_pairs is not None:
            return len(self.val_pairs)
        return len(self.scans)

    def _select_views(
        self, frame_ids: list[int], iou: torch.Tensor,
    ) -> tuple[list[int], list[int]]:
        """Select context and extra views, returning context IDs first."""
        items = frame_ids
        n = len(items)
        num_ctx = self.num_input_views
        num_extra = self.num_views - num_ctx

        if self.mode == "train":
            for _attempt in range(200):
                # Sample an anchor and overlap-compatible neighbors.
                idx1 = random.randint(0, n - 1)
                context_id1 = items[idx1]

                candidates = items[idx1 + self.min_frame_dist : idx1 + self.max_frame_dist + 1]

                candidates_stay = []
                for candidate in candidates:
                    overlap = iou[context_id1, candidate].item()
                    if self.iou_min < overlap < self.iou_max:
                        candidates_stay.append(candidate)

                need = (num_ctx - 1) + num_extra
                if len(candidates_stay) < need:
                    continue

                extra_ctx = random.sample(candidates_stay, num_ctx - 1)
                context_ids = sorted([context_id1] + extra_ctx)

                # Sample extra views from the context interval.
                idx_last = items.index(context_ids[-1])
                extra_pool = items[idx1 + 1 : idx_last + self.min_frame_dist + 1]
                extra_pool = [f for f in extra_pool if f not in context_ids]

                if len(extra_pool) < num_extra:
                    continue

                extra_views = random.sample(extra_pool, num_extra)

                all_ids = context_ids + sorted(extra_views)
                return context_ids, all_ids

            # Fall back to evenly spaced views.
            step = max(1, n // self.num_views)
            start = random.randint(0, max(0, n - step * self.num_views))
            indices = [min(start + i * step, n - 1) for i in range(self.num_views)]
            selected = [items[i] for i in indices]
            return selected[:num_ctx], selected
        else:
            # Evaluation uses deterministic spacing.
            step = max(1, n // self.num_views)
            indices = [min(i * step, n - 1) for i in range(self.num_views)]
            selected = [items[i] for i in indices]
            return selected[:num_ctx], selected

    @staticmethod
    def _preprocess_poses(
        c2ws: np.ndarray,
        scene_scale_factor: float = 1.35,
    ) -> np.ndarray:
        """Align camera poses to their average frame and normalize scale."""
        poses = c2ws.copy().astype(np.float64)

        # Estimate the shared camera frame.
        center = poses[:, :3, 3].mean(0)

        avg_forward = poses[:, :3, 2].mean(0)
        avg_forward = avg_forward / (np.linalg.norm(avg_forward) + 1e-8)

        avg_down = poses[:, :3, 1].mean(0)

        avg_right = np.cross(avg_down, avg_forward)
        avg_right = avg_right / (np.linalg.norm(avg_right) + 1e-8)

        avg_down = np.cross(avg_forward, avg_right)
        avg_down = avg_down / (np.linalg.norm(avg_down) + 1e-8)

        # Transform poses into the shared frame.
        avg_pose = np.eye(4, dtype=np.float64)
        avg_pose[:3, :3] = np.stack([avg_right, avg_down, avg_forward], axis=-1)
        avg_pose[:3, 3] = center
        avg_w2c = np.linalg.inv(avg_pose)

        poses = avg_w2c @ poses  # [V, 4, 4]

        # Normalize camera translation scale.
        scene_scale = np.abs(poses[:, :3, 3]).max()
        scene_scale = scene_scale_factor * max(scene_scale, 1e-6)
        poses[:, :3, 3] /= scene_scale

        return poses.astype(np.float32)

    def __getitem__(self, idx: int) -> dict:
        try:
            return self._getitem_inner(idx)
        except Exception as e:
            if isinstance(e, (ValueError, FileNotFoundError)):
                log.warning(f"[ScanNetRE10K] Error at idx {idx}: {e}")
                return self._getitem_inner((idx + 1) % len(self))
            raise

    def _getitem_inner(self, idx: int) -> dict:
        if self.val_pairs is not None:
            # Eval mode: use fixed pairs from val_pair.json
            pair = self.val_pairs[idx]
            scan_name = pair["scan"]
            scan_path = osp.join(self.scans_dir, scan_name)
            context_ids = pair["context_ids"]
            target_ids = pair["target_ids"]
            # Context first, then targets (dedup, context already in target_ids)
            all_ids = context_ids + [t for t in target_ids if t not in context_ids]
        else:
            # Train mode: IoU-based selection
            scan_name, frame_ids, iou = self.scans[idx]
            scan_path = osp.join(self.scans_dir, scan_name)
            context_ids, all_ids = self._select_views(frame_ids, iou)

        # --- Load intrinsics (single per scan) ---------------------------
        intrinsic_file = osp.join(scan_path, "intrinsic.txt")
        K_raw = np.loadtxt(intrinsic_file)
        if K_raw.shape[0] == 4:
            K_raw = K_raw[:3, :3]

        # Determine native image size from first frame
        first_img = Image.open(osp.join(scan_path, "color", f"{all_ids[0]}.jpg"))
        native_w, native_h = first_img.size

        # Scale intrinsics to target image_size
        sx = self.image_size / native_w
        sy = self.image_size / native_h
        fx = K_raw[0, 0] * sx
        fy = K_raw[1, 1] * sy
        cx = K_raw[0, 2] * sx
        cy = K_raw[1, 2] * sy

        # --- Load images and poses ---------------------------------------
        images = []
        c2ws = []
        valid_fids = []

        for fid in all_ids:
            # Pose (c2w)
            pose_file = osp.join(scan_path, "extrinsic", f"{fid}.txt")
            pose = np.loadtxt(pose_file)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                # Skip invalid, pad later
                continue

            # Image
            img_path = osp.join(scan_path, "color", f"{fid}.jpg")
            img = Image.open(img_path).convert("RGB")
            img = img.resize((self.image_size, self.image_size), Image.LANCZOS)
            img_np = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]

            images.append(img_np.transpose(2, 0, 1))  # [3, H, W]
            c2ws.append(pose)
            valid_fids.append(fid)

        # Pad if not enough valid views
        while len(images) < self.num_views:
            images.append(images[-1])
            c2ws.append(c2ws[-1])
            valid_fids.append(valid_fids[-1])

        # Trim to exact num_views
        images = images[: self.num_views]
        c2ws = c2ws[: self.num_views]
        valid_fids = valid_fids[: self.num_views]

        images = np.stack(images, axis=0)  # [V, 3, H, W]
        c2ws = np.stack(c2ws, axis=0)  # [V, 4, 4]

        # --- Preprocess poses (scene normalization) ----------------------
        c2ws = self._preprocess_poses(c2ws, self.scene_scale_factor)

        # --- Build fxfycxcy [V, 4] --------------------------------------
        fxfycxcy = np.array(
            [[fx, fy, cx, cy]] * self.num_views, dtype=np.float32,
        )

        result = {
            "image": torch.from_numpy(images).float(),
            "c2w": torch.from_numpy(c2ws).float(),
            "fxfycxcy": torch.from_numpy(fxfycxcy).float(),
            "scene_name": scan_name,
        }

        # --- Panoptic GT for context views (first num_input_views) -------
        if self.load_panoptic:
            context_fids = valid_fids[: self.num_input_views]
            # Load panoptic images for context views (uint8 RGB encoding)
            context_images_hw3 = [
                np.array(
                    Image.open(osp.join(scan_path, "color", f"{fid}.jpg")).convert("RGB")
                    .resize((self.image_size, self.image_size), Image.LANCZOS)
                )
                for fid in context_fids
            ]

            context_segm_raw = []
            for fid in context_fids:
                segm_path = osp.join(scan_path, "panoptic", f"{fid}.png")
                segm = np.array(Image.open(segm_path))
                # Resize if needed (nearest for label maps)
                if segm.shape[0] != self.image_size or segm.shape[1] != self.image_size:
                    segm = np.array(
                        Image.fromarray(segm).resize(
                            (self.image_size, self.image_size), Image.NEAREST,
                        )
                    )
                context_segm_raw.append(segm)

            # Decode panoptic PNG: R + G*256 + B*256*256
            context_segm = [
                s[:, :, 0].astype(np.int32)
                + s[:, :, 1].astype(np.int32) * 256
                + s[:, :, 2].astype(np.int32) * 256 * 256
                for s in context_segm_raw
            ]
            context_semantic = [s // 1000 for s in context_segm]
            context_instance = [s % 1000 for s in context_segm]

            # Build instance_id → semantic_id mapping per view
            segment_id2semantic_id = []
            for view_idx in range(len(context_fids)):
                semantic_labels = np.unique(context_semantic[view_idx])
                ins2sem = {}
                for sem_label in semantic_labels:
                    seg_ids = np.unique(
                        context_instance[view_idx][context_semantic[view_idx] == sem_label]
                    )
                    ins2sem.update({int(sid): int(sem_label) for sid in seg_ids})
                segment_id2semantic_id.append(ins2sem)

            # Mask2Former processor: produces mask_labels and class_labels
            m2f_inputs = self.processor.preprocess(
                video_frames=context_images_hw3,
                segmentation_maps=context_instance,
                instance_id_to_semantic_id=segment_id2semantic_id,
                return_tensors="pt",
            )
            result["context_mask_labels"] = m2f_inputs.mask_labels
            result["context_class_labels"] = m2f_inputs.class_labels

        # --- Panoptic GT for target views -------
        if self.load_target_panoptic:
            if self.target_has_input:
                target_fids = valid_fids  # all views (source included in targets)
            else:
                target_fids = valid_fids[self.num_input_views:]
            target_images_hw3 = [
                np.array(
                    Image.open(osp.join(scan_path, "color", f"{fid}.jpg")).convert("RGB")
                    .resize((self.image_size, self.image_size), Image.LANCZOS)
                )
                for fid in target_fids
            ]

            target_segm_raw = []
            for fid in target_fids:
                segm_path = osp.join(scan_path, "panoptic", f"{fid}.png")
                segm = np.array(Image.open(segm_path))
                if segm.shape[0] != self.image_size or segm.shape[1] != self.image_size:
                    segm = np.array(
                        Image.fromarray(segm).resize(
                            (self.image_size, self.image_size), Image.NEAREST,
                        )
                    )
                target_segm_raw.append(segm)

            # Decode panoptic PNG: R + G*256 + B*256*256
            target_segm = [
                s[:, :, 0].astype(np.int32)
                + s[:, :, 1].astype(np.int32) * 256
                + s[:, :, 2].astype(np.int32) * 256 * 256
                for s in target_segm_raw
            ]
            target_semantic = [s // 1000 for s in target_segm]
            target_instance = [s % 1000 for s in target_segm]

            segment_id2semantic_id_target = []
            for view_idx in range(len(target_fids)):
                semantic_labels = np.unique(target_semantic[view_idx])
                ins2sem = {}
                for sem_label in semantic_labels:
                    seg_ids = np.unique(
                        target_instance[view_idx][target_semantic[view_idx] == sem_label]
                    )
                    ins2sem.update({int(sid): int(sem_label) for sid in seg_ids})
                segment_id2semantic_id_target.append(ins2sem)

            m2f_target = self.processor.preprocess(
                video_frames=target_images_hw3,
                segmentation_maps=target_instance,
                instance_id_to_semantic_id=segment_id2semantic_id_target,
                return_tensors="pt",
            )
            result["target_mask_labels"] = m2f_target.mask_labels
            result["target_class_labels"] = m2f_target.class_labels

        return result


def scannet_re10k_collate_fn(batch: list[dict]) -> dict:
    """Custom collate for ScanNetRE10K batches with optional ragged mask/class labels.

    Standard fields (image, c2w, fxfycxcy) are stacked into tensors.
    Mask/class labels (different num_instances per sample) are kept as lists.
    """
    collated = {}

    # Ragged label keys that cannot be stacked
    list_keys = {
        "context_mask_labels", "context_class_labels",
        "target_mask_labels", "target_class_labels",
        "scene_name",
    }

    for key in batch[0].keys():
        values = [sample[key] for sample in batch]

        if key in list_keys:
            if key.endswith("_labels"):
                # Mask2Former expects list of B tensors, each [N_i, V, H, W].
                # The processor returns a 1-element list per sample; flatten it.
                grouped = []
                for sample_labels in values:
                    if isinstance(sample_labels, (list, tuple)):
                        grouped.extend(sample_labels)
                    else:
                        grouped.append(sample_labels)
                collated[key] = grouped
            else:
                collated[key] = values
        elif isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values)
        else:
            collated[key] = values

    return collated
