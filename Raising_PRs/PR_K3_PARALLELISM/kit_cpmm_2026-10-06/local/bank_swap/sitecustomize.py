"""Logbook kit only. With BANK_SWAP=1, the dynamic-CP tower computes the split bank without grad, then main's whole
bank, logs their difference and the grids, and hands main's bank to the language model. If the run then equals main
bitwise, the split path changes nothing but the bank's arithmetic."""

import os
import sys

if os.environ.get("BANK_SWAP") == "1" and "RANK" in os.environ:  # the torchrun parent has no RANK
    import torch

    from torchtitan.models.kimi_k3.vision_encoder import KimiK3VisionEncoder

    _orig = KimiK3VisionEncoder.forward
    _calls = [0]

    def _swap(self, pixel_values, *, grid_thw):
        subgroups = self._cp_subgroups
        if not subgroups:
            return _orig(self, pixel_values, grid_thw=grid_thw)
        with torch.no_grad():
            split = _orig(self, pixel_values, grid_thw=grid_thw)
        self._cp_subgroups = {}
        try:
            whole = _orig(self, pixel_values, grid_thw=grid_thw)
        finally:
            self._cp_subgroups = subgroups
        ref = whole.detach().float()
        diff = split.float() - ref
        line = (
            f"bank_swap: call {_calls[0]} grids {grid_thw.tolist()} bank {tuple(whole.shape)} {whole.dtype} "
            f"equal {torch.equal(split, whole.detach())} max_abs {diff.abs().max().item():.3e} "
            f"ref_max {ref.abs().max().item():.3e} rel_norm {(diff.norm() / ref.norm()).item():.3e}\n"
        )
        sys.stderr.write(line)
        if os.environ.get("BANK_SWAP_LOG"):
            with open(f"{os.environ['BANK_SWAP_LOG']}.rank{os.environ['RANK']}", "a") as f:
                f.write(line)
        _calls[0] += 1
        return whole

    KimiK3VisionEncoder.forward = _swap
    sys.stderr.write("bank_swap: KimiK3VisionEncoder.forward swapped\n")
