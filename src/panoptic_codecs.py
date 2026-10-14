"""Encode `sem*1000+inst` panoptic maps as three-channel NVS signals."""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


class PanopticCodec:
    name: str = "base"

    def encode(self, pano: Tensor) -> Tuple[Tensor, Dict]:
        """pano: [V, H, W] int64 (sem*1000 + inst). Returns (encoded, lut)."""
        raise NotImplementedError

    def decode(self, signal: Tensor, lut: Dict) -> Tensor:
        """signal: [V_tgt, 3, H, W] float in [0,1]. Returns pano [V_tgt, H, W] int64."""
        raise NotImplementedError


class BinaryCodec(PanopticCodec):
    """Encode labels as binary RGB codes and map uncertain pixels to zero."""

    name = "binary"

    # Robust code order for area-ranked instances; zero remains unlabeled.
    _MAX_SEP_ORDER_3BIT = [7, 4, 3, 5, 2, 6, 1]  # after 0 (unlabeled)

    def __init__(
        self,
        bits: int = 3,
        tau: float = 0.2,
        assign: str = "encounter",
        group: str = "instance",
    ):
        """Configure encounter/area assignment and instance/class grouping."""
        if bits < 1 or bits > 8:
            raise ValueError("bits must be in [1,8]")
        if assign not in ("encounter", "area"):
            raise ValueError(f"assign must be 'encounter' or 'area', got {assign!r}")
        if group not in ("instance", "class"):
            raise ValueError(f"group must be 'instance' or 'class', got {group!r}")
        self.bits = int(bits)
        self.tau = float(tau)
        self.assign = assign
        self.group = group
        self.K = 1 << self.bits  # total codewords (incl. 0 = unlabeled)

    def encode(self, pano: Tensor) -> Tuple[Tensor, Dict]:
        assert pano.dim() == 3, f"pano should be [V,H,W], got {tuple(pano.shape)}"
        V, H, W = pano.shape
        device = pano.device

        # Collapse instances when assigning one code per semantic class.
        if self.group == "class":
            sem = pano // 1000
            nonzero = sem != 0
            pano = torch.where(nonzero, sem * 1000 + 1, torch.zeros_like(pano))

        # Collect non-void labels across all source views.
        ids_all = torch.unique(pano)
        sem_all = ids_all // 1000
        keep = (ids_all != 0) & (sem_all != 0)
        instance_ids = ids_all[keep].tolist()  # list of sem*1000+inst >0

        # Keep the largest labels when the codebook is full.
        max_codewords = self.K - 1  # reserve code 0
        counts = {int(i): int((pano == i).sum().item()) for i in instance_ids}
        if len(instance_ids) > max_codewords:
            instance_ids = sorted(instance_ids, key=lambda k: counts[k], reverse=True)[
                :max_codewords
            ]

        # Select codewords by encounter or area order.
        if self.assign == "area":
            instance_ids = sorted(
                instance_ids, key=lambda k: counts[int(k)], reverse=True
            )
            if self.bits == 3:
                code_order = self._MAX_SEP_ORDER_3BIT
            else:
                code_order = list(
                    range(1, self.K)
                )  # no curated order for other bit widths
        else:  # "encounter"
            code_order = list(range(1, self.K))

        id_to_code: Dict[int, int] = {0: 0}  # unlabeled
        code_to_id: Dict[int, int] = {0: 0}
        for rank, iid in enumerate(instance_ids):
            code = code_order[rank]
            id_to_code[int(iid)] = code
            code_to_id[code] = int(iid)

        # Build integer codes with shape `[V, H, W]`.
        code_map = torch.zeros_like(pano, dtype=torch.int64)
        for iid, c in id_to_code.items():
            if iid == 0:
                continue
            code_map = torch.where(
                pano == iid, torch.tensor(c, dtype=torch.int64, device=device), code_map
            )

        # Expand codes into bit channels.
        chans = []
        for b in range(self.bits):
            chans.append(((code_map >> b) & 1).to(torch.float32))
        encoded = torch.stack(chans, dim=1)  # [V, bits, H, W]
        # Pad to the NVS model's three input channels.
        if self.bits < 3:
            pad = torch.zeros(
                V, 3 - self.bits, H, W, device=device, dtype=torch.float32
            )
            encoded = torch.cat([encoded, pad], dim=1)
        elif self.bits > 3:
            raise NotImplementedError("bits>3 requires >3 channels; NVS limited to 3.")

        return encoded, {"code_to_id": code_to_id, "bits": self.bits}

    def encode_multipass(self, pano: Tensor) -> Tuple[list, list]:
        """Partition labels into balanced passes and return `(signals, luts)`."""
        assert pano.dim() == 3, f"pano should be [V,H,W], got {tuple(pano.shape)}"
        V, H, W = pano.shape
        device = pano.device

        if self.group == "class":
            sem = pano // 1000
            nonzero = sem != 0
            pano = torch.where(nonzero, sem * 1000 + 1, torch.zeros_like(pano))

        ids_all = torch.unique(pano)
        sem_all = ids_all // 1000
        keep = (ids_all != 0) & (sem_all != 0)
        instance_ids = ids_all[keep].tolist()
        if not instance_ids:
            # Preserve one empty pass for an all-void input.
            single, lut = self.encode(pano)
            return [single], [lut]

        counts = {int(i): int((pano == i).sum().item()) for i in instance_ids}
        ordered = sorted(instance_ids, key=lambda k: -counts[int(k)])
        max_per_pass = self.K - 1  # reserve code 0
        n_passes = (len(ordered) + max_per_pass - 1) // max_per_pass

        # Round-robin area order balances large labels across passes.
        signals, luts = [], []
        for p in range(n_passes):
            sub_ids = ordered[p::n_passes]
            # Mask labels assigned to other passes.
            sub_pano = torch.zeros_like(pano)
            for iid in sub_ids:
                sub_pano = torch.where(
                    pano == iid,
                    torch.tensor(int(iid), dtype=pano.dtype, device=device),
                    sub_pano,
                )
            sig, lut = self.encode(sub_pano)
            signals.append(sig)
            luts.append(lut)
        return signals, luts

    def decode_merge(
        self, signals_decoded: list, masks: list = None, prefer: str = "area"
    ) -> Tensor:
        """Merge `[V_tgt,H,W]` passes, skipping optional uncertainty masks."""
        assert len(signals_decoded) >= 1
        merged = signals_decoded[0].clone()
        if masks is not None and masks[0] is not None:
            merged = torch.where(masks[0], torch.zeros_like(merged), merged)
        for p in range(1, len(signals_decoded)):
            cand = signals_decoded[p]
            if masks is not None and masks[p] is not None:
                cand = torch.where(masks[p], torch.zeros_like(cand), cand)
            if prefer == "first":
                fill = (merged == 0) & (cand != 0)
                merged = torch.where(fill, cand, merged)
            else:
                # Earlier non-zero labels win ties.
                fill = (merged == 0) & (cand != 0)
                merged = torch.where(fill, cand, merged)
        return merged

    def decode(self, signal: Tensor, lut: Dict, return_mask: bool = False):
        """Decode `[V,3,H,W]`; optionally return the threshold uncertainty mask."""
        assert (
            signal.dim() == 4 and signal.shape[1] == 3
        ), f"signal {tuple(signal.shape)}"
        bits = int(lut["bits"])
        code_to_id: Dict[int, int] = lut["code_to_id"]
        V, _, H, W = signal.shape
        device = signal.device

        # A pixel is uncertain when any used bit is near 0.5.
        used = signal[:, :bits]  # [V, bits, H, W]
        uncertain = (torch.abs(used - 0.5) <= self.tau).any(dim=1)  # [V, H, W]

        rounded = (used > 0.5).to(torch.int64)
        code = torch.zeros(V, H, W, dtype=torch.int64, device=device)
        for b in range(bits):
            code = code | (rounded[:, b] << b)

        # Map binary codes back to panoptic IDs.
        out = torch.zeros(V, H, W, dtype=torch.int64, device=device)
        for c, iid in code_to_id.items():
            if c == 0:
                continue
            out = torch.where(
                code == c, torch.tensor(int(iid), dtype=torch.int64, device=device), out
            )

        # Map uncertain pixels to void.
        out = torch.where(uncertain, torch.zeros_like(out), out)
        if return_mask:
            return out, uncertain
        return out


_CODECS = {
    "binary": BinaryCodec,
}


def make_codec(name: str, **kwargs) -> PanopticCodec:
    if name not in _CODECS:
        raise KeyError(f"Unknown codec {name!r}; known: {list(_CODECS)}")
    return _CODECS[name](**kwargs)


if __name__ == "__main__":
    # Smoke test
    pano = torch.tensor(
        [
            [[0, 1001, 1001, 2003], [0, 1001, 2003, 2003], [5012, 5012, 0, 0]],
            [[0, 1001, 2003, 2003], [0, 1001, 1001, 0], [5012, 5012, 0, 0]],
        ],
        dtype=torch.int64,
    )  # 2 views, 3x4
    codec = BinaryCodec(bits=3, tau=0.15)
    enc, lut = codec.encode(pano)
    print(
        f"encoded: {tuple(enc.shape)} range=[{enc.min().item():.2f},{enc.max().item():.2f}]"
    )
    print(f"lut: {lut}")
    # Verify a clean round trip.
    out = codec.decode(enc, lut)
    ok = torch.equal(out, pano)
    print(f"roundtrip ok: {ok}")
