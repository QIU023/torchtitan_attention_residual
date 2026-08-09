"""Why is grad_norm 0.0000 at step 1 under DEP + EP? Probe, do not guess.

Observed on `ep2 x fsdp2 x pp2` with `KIMI_VIT_DEP=1`, from a shared seed checkpoint:
step 1's loss is CORRECT (12.07418, identical to the DEP-off arm) and grad_norm is
**0.0000 on every rank**, then step 2 is NaN. Without EP the same configuration is
bit-identical to DEP off, so EP is a necessary condition.

`_clip_grad_norm_with_ep` skips a parameter when `p.grad is None`, so an all-zero norm is
what an empty grad set looks like from outside. This probe reports, per rank, how many
parameters exist, how many have a gradient, how many of those are non-finite, and which
mesh axes their DTensors live on -- the last one because that path branches on whether
"ep" appears in `p.device_mesh.mesh_dim_names`.

Entry point rather than an edit to core: it wraps
`torchtitan.distributed.utils.clip_grad_norm_` and then runs the normal trainer, so no
core file changes and the numbers come from the real training step.

Usage (mirrors any matrix cell, just swap the entry):
    KIMI_VIT_DEP=1 torchrun --nproc_per_node=4 \
      matrix_scripts/dep_ep_grad_probe.py --module kimi_k3 --config <flavor> ...
"""

import os

import torch

import torchtitan.distributed.utils as dist_utils

_orig = dist_utils.clip_grad_norm_


def _describe(parameters):
    # Report NORMS, not just finiteness. An earlier version of this probe reported only
    # "nonfinite_grads=0", which reads the same whether the gradients are healthy or
    # identically zero -- and zero is exactly what was suspected. A count that cannot
    # distinguish the two states measures nothing.
    total = have_grad = nonfinite = 0
    axes: dict[str, int] = {}
    not_dtensor = 0
    sq_ep = 0.0
    sq_non_ep = 0.0
    zero_grads = 0
    for p in parameters:
        total += 1
        if p.grad is None:
            continue
        have_grad += 1
        g = p.grad
        if isinstance(g, torch.distributed.tensor.DTensor):
            names = g.device_mesh.mesh_dim_names or ()
            key = "+".join(names) if names else "<unnamed>"
            axes[key] = axes.get(key, 0) + 1
            local = g.to_local()
        else:
            not_dtensor += 1
            local = g
        if not torch.isfinite(local).all():
            nonfinite += 1
        n2 = float(local.detach().double().pow(2).sum())
        if n2 == 0.0:
            zero_grads += 1
        if isinstance(g, torch.distributed.tensor.DTensor) and "ep" in (
            g.device_mesh.mesh_dim_names or ()
        ):
            sq_ep += n2
        else:
            sq_non_ep += n2
    return (
        total,
        have_grad,
        nonfinite,
        axes,
        not_dtensor,
        sq_ep**0.5,
        sq_non_ep**0.5,
        zero_grads,
    )


def probe(parameters, *args, **kwargs):
    params = list(parameters)
    total, have_grad, nonfinite, axes, not_dtensor, n_ep, n_non_ep, zeros = _describe(
        params
    )
    rank = int(os.environ.get("RANK", "-1"))
    out = _orig(params, *args, **kwargs)
    print(
        f"[grad-probe rank {rank}] params={total} with_grad={have_grad} "
        f"zero_grads={zeros} nonfinite={nonfinite} axes={axes} "
        f"local_norm_ep={n_ep:.4f} local_norm_non_ep={n_non_ep:.4f} "
        f"returned_total_norm={float(out):.4f}",
        flush=True,
    )
    return out


dist_utils.clip_grad_norm_ = probe

# train.py imports clip_grad_norm_ by module attribute in some versions and by name in
# others; patch both so the probe cannot be silently bypassed -- an unpatched run would
# look like a clean one.
import torchtitan.train as T  # noqa: E402

if hasattr(T, "dist_utils"):
    T.dist_utils.clip_grad_norm_ = probe
if hasattr(T, "clip_grad_norm_"):
    T.clip_grad_norm_ = probe

T.main()
