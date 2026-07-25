"""Measure the ACTUAL applied gradient scale under CP (2026-07-25).

Question: at cp>1 the trainer prints a grad_norm exactly 1/cp of the cp1
value. GAPS_TO_K3_SFT B9 called that cosmetic ("the metric prints the
local partial norm"). AdamW is scale-invariant, so no loss curve can
settle it -- a uniformly scaled gradient produces the same trajectory.

This probe runs the REAL trainer (real data, loss, CP input sharding,
FSDP reduction) and measures ``param.grad`` itself, bypassing
get_total_norm/_NormPartial entirely: every grad is densified with
full_tensor() and the global norm is summed by hand. If the hand-computed
norm also scales as 1/cp, the applied gradient really is under-scaled and
the effect is NOT cosmetic (it shifts the grad-clipping threshold and
breaks any scale-sensitive optimizer).

Usage (identical config, only cp differs):
  torchrun --nproc_per_node=1 cp_grad_scale_probe.py --module kimi_k3 \
    --config kimi_linear_debugmodel --checkpoint.no-enable \
    --debug.seed 42 --debug.deterministic --training.steps 1 \
    --parallelism.context_parallel_degree 1
  torchrun --nproc_per_node=2 ... --parallelism.context_parallel_degree 2
Compare the MANUAL_GRAD_NORM lines.
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import torchtitan.distributed.utils as dist_utils

_orig_clip = dist_utils.clip_grad_norm_


def _measuring_clip(parameters, *args, **kwargs):
    """Report the true norm of param.grad, then delegate unchanged."""
    params = (
        [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    )
    sq_total, n_grads = 0.0, 0
    for p in params:
        if p.grad is None:
            continue
        g = p.grad
        # full_tensor() reconstructs the unsharded gradient over every mesh
        # axis (fsdp = dp_shard x cp, tp, ...); a Replicate axis is a no-op.
        if isinstance(g, DTensor):
            g = g.full_tensor()
        sq_total += g.detach().float().pow(2).sum().item()
        n_grads += 1
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(
            f"[PROBE] MANUAL_GRAD_NORM {sq_total ** 0.5:.6f} "
            f"over {n_grads} grad tensors",
            flush=True,
        )
    return _orig_clip(parameters, *args, **kwargs)


dist_utils.clip_grad_norm_ = _measuring_clip
# trainer.py holds its own reference (`from ... import dist_utils` is a module
# import, so patching the module attribute above is enough), but be explicit:
import torchtitan.trainer as _trainer  # noqa: E402

if getattr(_trainer, "dist_utils", None) is not None:
    _trainer.dist_utils.clip_grad_norm_ = _measuring_clip

from torchtitan.train import main  # noqa: E402

if __name__ == "__main__":
    main()
