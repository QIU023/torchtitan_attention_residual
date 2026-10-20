"""H100 kit shims, logbook kit only: (1) torch 2.15.0.dev20260906 lacks MixedPrecisionPolicy.param_dtype_override_fn (main #5068),
accepted when None; (2) the #4380 bank swap probe, active only with BANK_SWAP=1."""

import sys

import torch.distributed.fsdp as _fsdp

_Policy = _fsdp.MixedPrecisionPolicy
if "param_dtype_override_fn" not in getattr(_Policy, "__dataclass_fields__", {}):

    def MixedPrecisionPolicy(*args, param_dtype_override_fn=None, **kwargs):
        if param_dtype_override_fn is not None:
            raise RuntimeError("mpp_shim: param_dtype_override_fn needs a newer torch")
        return _Policy(*args, **kwargs)

    _fsdp.MixedPrecisionPolicy = MixedPrecisionPolicy
    sys.stderr.write("mpp_shim: MixedPrecisionPolicy takes param_dtype_override_fn=None\n")


# --- torch 2.15.0.dev20260906 lacks two pipelining names main imports for DistMoE only (models/common/dist_moe/runtime.py) ---
import torch.distributed.pipelining as _pipelining

if not hasattr(_pipelining, "PipelineStageInfo"):
    _pipelining.PipelineStageInfo = type("PipelineStageInfo", (), {})
if not hasattr(_pipelining, "analyze_pipeline_activation_liveness"):

    def _liveness_needs_newer_torch(*args, **kwargs):
        raise RuntimeError("h100 shim: analyze_pipeline_activation_liveness needs a newer torch")

    _pipelining.analyze_pipeline_activation_liveness = _liveness_needs_newer_torch

# --- bank swap (kit_cpmm_2026-10-06/local/bank_swap) ---
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
