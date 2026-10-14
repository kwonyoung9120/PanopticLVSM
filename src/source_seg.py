"""Oracle and PanSt3R source segmenters returning `sem*1000+inst` maps."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import torch
from torch import Tensor

# ScanNet 21-class taxonomy (0 = unlabeled, 1..20 valid).
_SCANNET_NAMES = [
    "unlabeled",
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]

# PanSt3R ADE20K vocabulary.
_PANST3R_ADE20K_VOCAB = [
    "clothes",
    "basket, handbasket",
    "armchair",
    "painting, picture",
    "cradle",
    "booth",
    "bicycle",
    "barrel, cask",
    "rug",
    "desk",
    "street lamp",
    "food, solid food",
    "stove",
    "toilet, can, commode, crapper, pot, potty, stool, throne",
    "tent",
    "escalator, moving staircase, moving stairway",
    "sofa",
    "table",
    "swivel chair",
    "minibike, motorbike",
    "ship",
    "hovel, hut, hutch, shack, shanty",
    "stage",
    "bottle",
    "van",
    "wardrobe, closet, press",
    "mountain, mount",
    "towel",
    "hill",
    "blanket, cover",
    "trash can",
    "field",
    "computer",
    "chest of drawers, chest, bureau, dresser",
    "sand",
    "lake",
    "fence",
    "flower",
    "fan",
    "road, route",
    "pillow",
    "sink",
    "coffee table",
    "wall",
    "lamp",
    "bridge, span",
    "pool table, billiard table, snooker table",
    "grandstand, covered stand",
    "rock, stone",
    "bag",
    "bookcase",
    "sea",
    "ottoman, pouf, pouffe, puff, hassock",
    "truck",
    "building",
    "tv",
    "light",
    "glass, drinking glass",
    "ball",
    "falls",
    "palm, palm tree",
    "oven",
    "blind, screen",
    "mirror",
    "bar",
    "stairway, staircase",
    "poster, posting, placard, notice, bill, card",
    "awning, sunshade, sunblind",
    "curtain",
    "fireplace",
    "animal",
    "bannister, banister, balustrade, balusters, handrail",
    "earth, ground",
    "boat",
    "bench",
    "canopy",
    "counter",
    "hood, exhaust hood",
    "bulletin board",
    "clock",
    "case, display case, showcase, vitrine",
    "tree",
    "base, pedestal, stand",
    "book",
    "flag",
    "tower",
    "runway",
    "trade name",
    "plate",
    "skyscraper",
    "column, pillar",
    "sconce",
    "car",
    "box",
    "stairs",
    "crt screen",
    "river",
    "shower",
    "buffet, counter, sideboard",
    "grass",
    "pier",
    "person",
    "chair",
    "pool",
    "water",
    "kitchen island",
    "plant",
    "plane",
    "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
    "cabinet",
    "land, ground, soil",
    "traffic light",
    "fountain",
    "tank, storage tank",
    "door",
    "ceiling",
    "shelf",
    "cushion",
    "floor",
    "screen door, screen",
    "plaything, toy",
    "refrigerator, icebox",
    "step, stair",
    "microwave",
    "countertop",
    "signboard, sign",
    "path",
    "arcade machine",
    "vase",
    "washer, automatic washer, washing machine",
    "rail",
    "radiator",
    "sky",
    "dishwasher",
    "tray",
    "tub",
    "window ",
    "screen",
    "sidewalk, pavement",
    "monitor",
    "stool",
    "house",
    "seat",
    "dirt track",
    "pot",
    "sculpture",
    "bus",
    "pole",
    "chandelier",
    "bed",
]


# PanSt3R's ScanNet vocab used at precompute time (from demo_panst3r.py).
_PANST3R_SCANNET_VOCAB = [
    "backpack",
    "printer",
    "paper bag",
    "shoe rack",
    "heater",
    "bowl",
    "power strip",
    "sofa",
    "table",
    "bottle",
    "telephone",
    "crate",
    "toilet brush",
    "towel",
    "trash can",
    "ceiling lamp",
    "toilet paper",
    "stapler",
    "plant pot",
    "picture",
    "pillow",
    "sink",
    "wall",
    "pan",
    "bag",
    "storage cabinet",
    "paper towel",
    "jacket",
    "exhaust fan",
    "cup",
    "tv",
    "paper",
    "blind rail",
    "binder",
    "basket",
    "container",
    "kitchen cabinet",
    "curtain",
    "socket",
    "refrigerator",
    "table lamp",
    "coat hanger",
    "bookshelf",
    "soap dispenser",
    "doorframe",
    "clock",
    "speaker",
    "blinds",
    "office chair",
    "slippers",
    "jar",
    "book",
    "cutting board",
    "laptop",
    "tissue box",
    "air vent",
    "kitchen counter",
    "box",
    "bucket",
    "spray bottle",
    "computer tower",
    "kettle",
    "marker",
    "cloth",
    "mouse",
    "smoke detector",
    "clothes hanger",
    "chair",
    "plant",
    "cabinet",
    "shoes",
    "poster",
    "door",
    "ceiling",
    "keyboard",
    "shelf",
    "cushion",
    "floor",
    "painting",
    "microwave",
    "toilet",
    "window",
    "tap",
    "board",
    "whiteboard",
    "blanket",
    "whiteboard eraser",
    "monitor",
    "headphones",
    "light switch",
    "windowsill",
    "pot",
    "window frame",
    "suitcase",
    "electrical duct",
    "pipe",
    "rack",
    "file folder",
    "bed",
    "shower wall",
]

_PANST3R_SYNONYMS = {
    # --- ScanNet++ 100 vocab synonyms (scannetpp100) ---
    "shelf": "bookshelf",
    "painting": "picture",
    "poster": "picture",
    "office chair": "chair",
    "doorframe": "door",
    "window frame": "window",
    "windowsill": "window",
    "kitchen cabinet": "cabinet",
    "storage cabinet": "cabinet",
    "blind rail": "curtain",
    "blinds": "curtain",
    "shower wall": "shower curtain",
    "sh. curtain": "shower curtain",
    "sh curtain": "shower curtain",
    "kitchen counter": "counter",
    "fridge": "refrigerator",
    "blanket": "bed",
    "trash can": "otherfurniture",
    "plant pot": "otherfurniture",
    "table lamp": "otherfurniture",
    "ceiling lamp": "otherfurniture",
    # --- ADE20K vocab synonyms (ade20k) — ADE20K labels include
    # comma-separated alt names; keep both the comma-joined full string
    # and the first token so remap lookup works either way. ---
    "painting, picture": "picture",
    "basket, handbasket": "otherfurniture",
    "blanket, cover": "bed",
    "armchair": "chair",
    "swivel chair": "chair",
    "ottoman, pouf, pouffe, puff, hassock": "sofa",
    "chest of drawers, chest, bureau, dresser": "cabinet",
    "wardrobe, closet, press": "cabinet",
    "coffee table": "table",
    "kitchen island": "counter",
    "buffet, counter, sideboard": "counter",
    "countertop": "counter",
    "bookcase": "bookshelf",
    "refrigerator, icebox": "refrigerator",
    "hood, exhaust hood": "cabinet",  # kitchen hood → cabinet (ScanNet GT convention)
    "toilet, can, commode, crapper, pot, potty, stool, throne": "toilet",
    "stove": "otherfurniture",
    "oven": "otherfurniture",
    "dishwasher": "otherfurniture",
    "washer, automatic washer, washing machine": "otherfurniture",
    "blind, screen": "curtain",
    "signboard, sign": "picture",
    "case, display case, showcase, vitrine": "cabinet",
    "poster, posting, placard, notice, bill, card": "picture",
    "window ": "window",  # note trailing space from ADE20K list
    "window-other": "window",
    "lamp": "otherfurniture",
    "mirror": "picture",
    "awning, sunshade, sunblind": "curtain",
    "shower": "shower curtain",
    "tub": "bathtub",
    # NOTE: classes that are ≥75% GT-unlabeled go in _PANST3R_TO_UNLABELED.
}

# Data-driven: PanSt3R classes whose pixels are predominantly ScanNet-
# GT-unlabeled (≥ 80%, measured on 50-scene val subset). Mapping these
# to class 0 avoids polluting otherfurniture FPs.
_PANST3R_TO_UNLABELED = {
    "ceiling",  # 99%
    "pipe",  # 99%
    "backpack",  # 99%
    "cloth",  # 99%
    "suitcase",  # 98%
    "towel",  # 98%
    "cushion",  # 95%
    "jar",  # 95%
    "bottle",  # 95%
    "basket",  # 96%
    "cup",  # 91%
    "tv",  # 91% (on walls, not annotated)
    "printer",  # 92%
    "microwave",  # 90%
    "box",  # 90%
    "monitor",  # 89%
    "plant",  # 88%
    "kettle",  # 84%
    "paper towel",  # 81%
    # wall-mounted fixtures — never annotated in ScanNet GT
    "light switch",
    "socket",
    "air vent",
    "exhaust fan",
    "smoke detector",
    "electrical duct",
    "clock",
    "whiteboard eraser",
}


class OracleSourceSeg:
    name = "oracle"

    def __call__(
        self,
        source_rgb: Tensor,  # [V, 3, H, W] (unused — oracle reads GT)
        source_panoptic: Tensor,  # [V, H, W] int64 sem*1000+inst
        scene: str,
        context_ids: List[int],
    ) -> dict:
        return {"pano": source_panoptic.clone(), "source": "oracle"}


class PanSt3RSourceSeg:
    """Run lazy PanSt3R inference and remap predictions to ScanNet-20."""

    name = "panst3r"

    def __init__(
        self,
        ckpt_path: str = "ckpts/panst3r_v2_512_5ds.pth",
        inference_size: int = 512,
        output_size: int = 256,
        device: str = "cuda",
        postprocess: str = "standard_v2",
        remap_unknown: str = "otherfurniture",  # "otherfurniture" | "unlabeled"
        vocab: str = "scannetpp100",  # "scannetpp100" | "scannet20"
    ):
        self.ckpt_path = ckpt_path
        self.inference_size = int(inference_size)
        self.output_size = int(output_size)
        self.device = device
        self.postprocess = postprocess
        self.remap_unknown = remap_unknown
        self.vocab_name = vocab
        self._model = None  # lazy
        if vocab == "scannetpp100":
            self._vocab = _PANST3R_SCANNET_VOCAB
            self._synonyms = _PANST3R_SYNONYMS
        elif vocab == "ade20k":
            self._vocab = _PANST3R_ADE20K_VOCAB
            self._synonyms = _PANST3R_SYNONYMS
        elif vocab == "scannetpp_ade":
            # Query the combined ScanNet++ and ADE20K vocabulary.
            self._vocab = _PANST3R_SCANNET_VOCAB + _PANST3R_ADE20K_VOCAB
            self._synonyms = _PANST3R_SYNONYMS
        elif vocab == "scannet20":
            self._vocab = [n for n in _SCANNET_NAMES if n != "unlabeled"]
            self._synonyms = {}
        elif vocab == "scannet20_fix":
            # Use a natural-language prompt for other furniture.
            base = [n for n in _SCANNET_NAMES if n != "unlabeled"]
            self._vocab = []
            for n in base:
                if n == "otherfurniture":
                    self._vocab.append("miscellaneous furniture")
                else:
                    self._vocab.append(n)
            self._synonyms = {
                "miscellaneous furniture": "otherfurniture",
            }
        elif vocab == "scannet20_of":
            # Alternate spelling for other furniture.
            base = [n for n in _SCANNET_NAMES if n != "unlabeled"]
            self._vocab = [
                ("other furniture" if n == "otherfurniture" else n) for n in base
            ]
            self._synonyms = {"other furniture": "otherfurniture"}
        elif vocab == "scannet20_hybrid":
            # SN-20 + concrete object classes that tend to be labeled
            # "otherfurniture" in ScanNet GT. The extras remap to class
            # 20 at decode. Keeps SN-20 accuracy on main classes while
            # giving SigLIP specific English words for the grab-bag
            # class 20.
            base = [n for n in _SCANNET_NAMES if n != "unlabeled"]
            base = [("other furniture" if n == "otherfurniture" else n) for n in base]
            extras = [
                "trash can",
                "box",
                "basket",
                "tv",
                "monitor",
                "microwave",
                "printer",
                "lamp",
                "stove",
                "oven",
                "dishwasher",
                "plant pot",
                "table lamp",
                "ceiling lamp",
            ]
            self._vocab = base + extras
            self._synonyms = {
                "other furniture": "otherfurniture",
                **{e: "otherfurniture" for e in extras},
            }
        elif vocab == "scannet20_rich2":
            # Data-driven pruned hybrid: keep only proxies that net-gained
            # PQ on full val in the scannet20_rich sweep. Removes
            # {pillow, blanket, ottoman, shelf/bookcase, painting/poster,
            # window/door over-generic} — they added more FPs than TPs.
            base = [n for n in _SCANNET_NAMES if n != "unlabeled"]
            base = [("other furniture" if n == "otherfurniture" else n) for n in base]
            extra_map = {
                # otherfurniture concrete objects (big win: +0.264 PQ)
                "trash can": "otherfurniture",
                "box": "otherfurniture",
                "basket": "otherfurniture",
                "tv": "otherfurniture",
                "monitor": "otherfurniture",
                "microwave": "otherfurniture",
                "printer": "otherfurniture",
                "lamp": "otherfurniture",
                "stove": "otherfurniture",
                "oven": "otherfurniture",
                "dishwasher": "otherfurniture",
                "plant pot": "otherfurniture",
                "table lamp": "otherfurniture",
                "ceiling lamp": "otherfurniture",
                # desk (net +0.011)
                "office desk": "desk",
                "study desk": "desk",
                "writing desk": "desk",
                "workbench": "desk",
                # counter (net +0.037)
                "kitchen counter": "counter",
                "bathroom counter": "counter",
                "countertop": "counter",
                "bar counter": "counter",
                # chair (net +0.012)
                "armchair": "chair",
                "office chair": "chair",
                "dining chair": "chair",
                # cabinet (net +0.005)
                "wardrobe": "cabinet",
                "dresser": "cabinet",
                "kitchen cabinet": "cabinet",
                "storage cabinet": "cabinet",
                # table (net −0.001, keep — might gain elsewhere)
                "coffee table": "table",
                "dining table": "table",
                "side table": "table",
                # refrigerator (same)
                "fridge": "refrigerator",
                # curtain (+0.003)
                "drape": "curtain",
                "blind": "curtain",
                # sink (+0.024)
                # (no new extras for sink per se, just reusing default)
                # keep toilet / bathtub as-is
                "other furniture": "otherfurniture",
            }
            self._vocab = list(dict.fromkeys(base + list(extra_map.keys())))
            self._synonyms = extra_map
        elif vocab == "scannet20_rich":
            # SN-20 with concrete English proxies for many weak classes.
            # All extras remap back to a canonical SN-20 label.
            base = [n for n in _SCANNET_NAMES if n != "unlabeled"]
            base = [("other furniture" if n == "otherfurniture" else n) for n in base]
            extra_map = {
                # otherfurniture grab-bag (from hybrid)
                "trash can": "otherfurniture",
                "box": "otherfurniture",
                "basket": "otherfurniture",
                "tv": "otherfurniture",
                "monitor": "otherfurniture",
                "microwave": "otherfurniture",
                "printer": "otherfurniture",
                "lamp": "otherfurniture",
                "stove": "otherfurniture",
                "oven": "otherfurniture",
                "dishwasher": "otherfurniture",
                "plant pot": "otherfurniture",
                "table lamp": "otherfurniture",
                "ceiling lamp": "otherfurniture",
                # desk proxies
                "office desk": "desk",
                "study desk": "desk",
                "writing desk": "desk",
                "workbench": "desk",
                # window proxies
                "glass window": "window",
                "windowpane": "window",
                "window frame": "window",
                "windowsill": "window",
                # counter proxies
                "kitchen counter": "counter",
                "bathroom counter": "counter",
                "countertop": "counter",
                "bar counter": "counter",
                # bookshelf proxies
                "shelf": "bookshelf",
                "bookcase": "bookshelf",
                "book shelf": "bookshelf",
                "bookstand": "bookshelf",
                # door proxies
                "doorway": "door",
                "doorframe": "door",
                "entry door": "door",
                "wooden door": "door",
                # picture proxies
                "painting": "picture",
                "poster": "picture",
                "photo frame": "picture",
                "framed picture": "picture",
                # chair proxies
                "armchair": "chair",
                "office chair": "chair",
                "dining chair": "chair",
                # sofa proxies
                "couch": "sofa",
                "ottoman": "sofa",
                # cabinet proxies
                "wardrobe": "cabinet",
                "dresser": "cabinet",
                "kitchen cabinet": "cabinet",
                "storage cabinet": "cabinet",
                # table proxies
                "coffee table": "table",
                "dining table": "table",
                "side table": "table",
                # bed / bedding
                "blanket": "bed",
                "pillow": "bed",
                # shower curtain
                "shower wall": "shower curtain",
                # refrigerator
                "fridge": "refrigerator",
                # curtain
                "drape": "curtain",
                "blind": "curtain",
                # name fix
                "other furniture": "otherfurniture",
            }
            self._vocab = list(dict.fromkeys(base + list(extra_map.keys())))
            self._synonyms = extra_map
        elif vocab == "scannet20_plus":
            # Query 20 ScanNet classes PLUS a small set of "helper" classes
            # that PanSt3R reliably picks but ScanNet labels as unlabeled.
            self._vocab = [n for n in _SCANNET_NAMES if n != "unlabeled"] + [
                "ceiling",
                "pipe",
                "light switch",
            ]
            self._synonyms = {
                "ceiling": "unlabeled",
                "pipe": "unlabeled",
                "light switch": "unlabeled",
            }
        elif vocab == "scannet20_multi":
            # Query several synonyms per ScanNet-20 class so SigLIP has
            # a better chance of matching. Everything remaps back to a
            # canonical ScanNet-20 label at decode time.
            self._vocab = [
                "wall",
                "floor",
                "cabinet",
                "cupboard",
                "bed",
                "chair",
                "office chair",
                "sofa",
                "couch",
                "table",
                "coffee table",
                "dining table",
                "door",
                "doorframe",
                "window",
                "bookshelf",
                "shelf",
                "picture",
                "painting",
                "poster",
                "counter",
                "countertop",
                "desk",
                "curtain",
                "refrigerator",
                "fridge",
                "shower curtain",
                "toilet",
                "sink",
                "bathtub",
                "otherfurniture",
                "ceiling",
                "pipe",
                "light switch",
            ]
            self._synonyms = {
                "cupboard": "cabinet",
                "office chair": "chair",
                "couch": "sofa",
                "coffee table": "table",
                "dining table": "table",
                "doorframe": "door",
                "shelf": "bookshelf",
                "painting": "picture",
                "poster": "picture",
                "countertop": "counter",
                "fridge": "refrigerator",
                "ceiling": "unlabeled",
                "pipe": "unlabeled",
                "light switch": "unlabeled",
            }
        else:
            raise ValueError(f"unknown vocab {vocab!r}")

    def _ensure_model(self):
        if self._model is not None:
            return
        import sys

        _HERE = str(Path(__file__).resolve().parent)
        if _HERE not in sys.path:
            sys.path.insert(0, _HERE)
        _REPO_ROOT = Path(__file__).resolve().parent.parent
        _CANDIDATES = [
            os.environ.get("PANST3R_SRC"),
            os.environ.get("MUST3R_SRC"),
            str(_REPO_ROOT / "seg_models" / "must3r"),
            str(_REPO_ROOT / "seg_models" / "panst3r" / "src"),
            str(_REPO_ROOT / "seg_models" / "panst3r"),
        ]
        for p in _CANDIDATES:
            if p and Path(p).exists() and p not in sys.path:
                sys.path.insert(0, p)
        from panst3r.panst3r import PanSt3R

        print(
            f"[PanSt3R] loading from {self.ckpt_path}  vocab={self.vocab_name}  infer_size={self.inference_size}",
            flush=True,
        )
        m = PanSt3R.from_checkpoint(self.ckpt_path).to(self.device).eval()
        for p in m.parameters():
            p.requires_grad = False
        m.panoptic_decoder.text_encoder.set_vocab(self._vocab, device=self.device)
        self._model = m

    def _resize_masks_to_output(self, pan_list):
        import numpy as np, cv2

        if self.inference_size == self.output_size:
            return pan_list
        out = []
        for pan in pan_list:
            pan_np = pan if isinstance(pan, np.ndarray) else pan.cpu().numpy()
            pan_np = pan_np.astype(np.int32)
            pan_np = cv2.resize(
                pan_np,
                (self.output_size, self.output_size),
                interpolation=cv2.INTER_NEAREST,
            )
            out.append(pan_np)
        return out

    def _remap(self, blob: dict) -> torch.Tensor:
        """Map raw PanSt3R masks and categories to ScanNet panoptic IDs."""
        import numpy as np

        pan_list = blob.get("pred_masks")
        segs_info = blob.get("segments_info", [])
        if pan_list is None:
            return blob["pano"]  # legacy cache with only pano
        if not segs_info:
            # Return void labels when no segments are found.
            h = w = self.output_size
            return torch.zeros(len(pan_list), h, w, dtype=torch.int64)

        seen = {}
        per_class_counter = {}
        for seg in segs_info:
            cls_name = self._vocab[seg["category_id"]].lower()
            if cls_name in _SCANNET_NAMES:
                sem = _SCANNET_NAMES.index(cls_name)
            elif cls_name in self._synonyms:
                sem = _SCANNET_NAMES.index(self._synonyms[cls_name])
            elif cls_name in _PANST3R_TO_UNLABELED:
                # Map classes dominated by ScanNet void pixels to class 0.
                sem = 0
            else:
                if self.remap_unknown == "otherfurniture":
                    sem = 20
                elif self.remap_unknown in ("unlabeled", "drop"):
                    sem = 0
                else:
                    raise ValueError(f"unknown remap_unknown={self.remap_unknown!r}")
            if sem == 0:
                seen[seg["id"]] = (0, 0)
                continue
            per_class_counter[sem] = per_class_counter.get(sem, 0) + 1
            seen[seg["id"]] = (sem, per_class_counter[sem])

        out_list = []
        for v in range(len(pan_list)):
            pan = pan_list[v]
            if not isinstance(pan, np.ndarray):
                pan = pan.cpu().numpy()
            out = np.zeros_like(pan, dtype=np.int64)
            for pid, (sem, inst) in seen.items():
                if sem == 0:
                    continue
                out[pan == pid] = sem * 1000 + inst
            out_list.append(torch.from_numpy(out))
        return torch.stack(out_list, dim=0)

    @torch.no_grad()
    def infer_batch(self, pairs: list) -> list:
        """Process `B` pairs and return one panoptic result per pair."""
        self._ensure_model()
        from panst3r.engine.postprocess import (
            panoptic_inference_v1,
            panoptic_inference_v2,
            panoptic_inference_qubo,
        )

        B = len(pairs)
        if B == 0:
            return []

        # Stack views, then restore `[B,V,3,H,W]` for batched attention.
        src_all = torch.cat([p.context_images for p in pairs], dim=0)  # [B*V, 3, H, W]
        V_per = pairs[0].context_images.shape[0]

        if (src_all.shape[-1], src_all.shape[-2]) != (
            self.inference_size,
            self.inference_size,
        ):
            src_all = torch.nn.functional.interpolate(
                src_all,
                size=(self.inference_size, self.inference_size),
                mode="bilinear",
                align_corners=False,
            )
        imgs_dv = (src_all.to(self.device) - 0.5) * 2.0  # [B*V, 3, H, W] dust3r norm
        imgs = imgs_dv.reshape(B, V_per, 3, self.inference_size, self.inference_size)
        true_shape = (
            torch.tensor(
                [[self.inference_size, self.inference_size]] * V_per,
                dtype=torch.int32,
                device=self.device,
            )
            .unsqueeze(0)
            .expand(B, V_per, 2)
            .contiguous()
        )  # [B, V, 2]
        # Keep outputs on GPU for postprocessing.
        out_pan, _ = self._model.forward(
            imgs,
            true_shape,
            self._vocab,
            max_bs=V_per,
            outdevice=None,
        )

        if self.postprocess == "qubo":
            fn = panoptic_inference_qubo
        elif self.postprocess == "standard_v1":
            fn = panoptic_inference_v1
        else:
            fn = panoptic_inference_v2

        label_mode = self._model.panoptic_decoder.label_mode

        # Postprocess each scene independently.
        pred_masks = out_pan["pred_masks"]
        pred_logits = out_pan["pred_logits"]
        assert (
            isinstance(pred_masks, torch.Tensor) and pred_masks.dim() == 5
        ), f"expected [B,V,Q,h,w] pred_masks, got {pred_masks.shape}"
        assert (
            pred_logits.dim() == 3
        ), f"expected [B,Q,K] pred_logits, got {pred_logits.shape}"

        out_list = []
        for b in range(B):
            masks_b = [
                pred_masks[b, v].unsqueeze(0) for v in range(V_per)
            ]  # list[[1,Q,h,w]]
            logits_b = pred_logits[b : b + 1]  # [1, Q, K]
            true_shape_b = true_shape[b].cpu().numpy()  # [V, 2]
            # GPU postprocessing avoids per-query CPU loops.
            pan_preds_b = fn(
                logits_b,
                masks_b,
                true_shape_b,
                label_mode=label_mode,
                device=self.device,
                multi_ar=True,
            )
            pan_list = pan_preds_b[0]["pan"]
            segs_info = pan_preds_b[0]["segments_info"]
            # Convert masks to NumPy for remapping.
            import numpy as np

            pan_list = [
                (
                    p.cpu().numpy()
                    if isinstance(p, torch.Tensor)
                    else (p if isinstance(p, np.ndarray) else p)
                )
                for p in pan_list
            ]
            pan_list = self._resize_masks_to_output(pan_list)
            blob = {
                "pano": None,
                "pred_masks": pan_list,
                "segments_info": segs_info,
                "size": self.output_size,
                "infer_size": self.inference_size,
                "source": "panst3r_batched",
            }
            out_list.append({"pano": self._remap(blob), "source": "panst3r_batched"})
        return out_list

    @torch.no_grad()
    def __call__(
        self,
        source_rgb: Tensor,  # [V, 3, H, W] in [0, 1]
        source_panoptic: Tensor,  # ignored
        scene: str,
        context_ids: List[int],
    ) -> dict:
        # Reuse the batched forward path for a single pair.
        from types import SimpleNamespace

        pair = SimpleNamespace(context_images=source_rgb)
        outs = self.infer_batch([pair])
        return {"pano": outs[0]["pano"], "source": "panst3r_inline"}


def make_source_seg(cfg, device: str = "cuda", output_size: int = 256):
    """Build the configured oracle, PanSt3R, SAM 2, or shared-query segmenter."""
    if hasattr(cfg, "name"):
        params = {k: v for k, v in dict(cfg).items() if k != "name"}
        name = cfg.name
    else:  # plain string
        params, name = {}, cfg
    if name == "oracle":
        return OracleSourceSeg()
    from config import resolve_ckpt

    params.setdefault("output_size", output_size)
    params.setdefault("device", device)
    if "ckpt" in params:
        params["ckpt_path"] = resolve_ckpt(params.pop("ckpt"))
    if name == "panst3r":
        return PanSt3RSourceSeg(**params)
    if name == "sam2":
        from sam2_seg import SAM2SourceSeg

        return SAM2SourceSeg(**params)
    if name == "shared_query":
        from shared_query_seg import SharedQuerySourceSeg

        return SharedQuerySourceSeg(**params)
    raise KeyError(f"Unknown source_seg {name!r}")
