"""Which Replicate-on-TP gradients disagree across ranks, and from where.

Written as a constraint because the two previous "findings" here were the same
measurement error twice: comparing values that are SUPPOSED to differ per rank
(a Shard-placed local, and a plain per-rank tensor such as o_proj's input, which
is the local attention-head shard). This probe therefore refuses to compare
anything that is not labelled Replicate on the TP axis.

A gradient labelled Replicate on the TP axis must hold identical values on every
TP rank; that is what the label asserts. A disagreement there is a real defect --
something turned a Partial into a Replicate without reducing.

Deliberately run on a DENSE flavor: MoE top-k is a discrete choice, so on a MoE
model any numerical difference eventually flips which expert a token reaches and
a cross-rank comparison measures route divergence instead of correctness.

    torchrun --nproc_per_node=2 -m ... --module kimi_k3 \
      --config kimi_k3_mini_diag_4l_mla_lora --parallelism.tensor_parallel_degree 2

with this file as the entry point (it wraps the trainer, then delegates to
torchtitan.train.main).
"""

from __future__ import annotations

import json
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

RANK = int(os.environ.get("RANK", "0"))
# Always per-rank. A shared path makes every finding appear twice, which reads
# as two findings.
_BASE = os.environ.get("LORA_PROBE_OUT", "/tmp/lora_repl.jsonl")
OUT = _BASE.replace(".jsonl", f"_r{RANK}.jsonl")


def _record(**kw) -> None:
    with open(OUT, "a") as f:
        f.write(json.dumps(kw) + "\n")
        f.flush()


def _tp_axis(mesh) -> int | None:
    names = mesh.mesh_dim_names
    if not names:
        return None
    for want in ("tp", "tensor_parallel"):
        if want in names:
            return names.index(want)
    return None


def _is_replicate_on_tp(t: torch.Tensor) -> bool:
    """True only for a DTensor whose TP-axis placement is Replicate.

    Everything else -- plain tensors, Shard on the TP axis, Partial, a mesh with
    no TP axis -- is NOT comparable across TP ranks and is excluded.
    """
    if not isinstance(t, DTensor):
        return False
    axis = _tp_axis(t.device_mesh)
    if axis is None:
        return False
    return isinstance(t.placements[axis], Replicate)


def compare_replicated_grads(model_parts, step: int) -> None:
    """All-gather every Replicate-on-TP gradient and report disagreements."""
    group = None
    checked = 0
    for part in model_parts:
        for name, p in part.named_parameters(remove_duplicate=False):
            g = p.grad
            # Record WHY a parameter was not compared. Absence from the
            # disagreement list otherwise reads as "clean" when it may mean
            # "never compared" -- the exact confusion that produced two false
            # conclusions here already.
            if g is None:
                _record(step=step, param=name, skipped="no_grad")
                continue
            if not _is_replicate_on_tp(g):
                _record(
                    step=step,
                    param=name,
                    skipped="not_replicate_on_tp",
                    placements=(
                        [str(x) for x in g.placements]
                        if isinstance(g, DTensor)
                        else "plain"
                    ),
                )
                continue
            checked += 1
            local = g.to_local().detach().float()
            gathered = [torch.empty_like(local) for _ in range(dist.get_world_size(group))]
            dist.all_gather(gathered, local, group=group)
            ref = gathered[0]
            max_delta = max(
                (other - ref).abs().max().item() for other in gathered[1:]
            )
            # Record agreements too. Recording only disagreements makes silence
            # mean "compared and agreed" in one place and "never compared" in
            # another, which is the ambiguity this probe exists to remove -- and
            # reading an agreement as an absence has already produced two wrong
            # conclusions here.
            _record(
                step=step,
                param=name,
                agrees=(max_delta == 0.0),
                max_delta=max_delta,
                ref_absmax=ref.abs().max().item(),
                placements=[str(x) for x in g.placements],
            )
    _record(step=step, summary=True, replicated_grads_checked=checked)


def dump_structure(model_parts, only: str) -> None:
    """What IS this module, before theorising about its gradient.

    Records the class, every parameter with its placement and requires_grad, and
    every child, for any module whose qualified name contains ``only``.
    """
    for part in model_parts:
        for name, mod in part.named_modules():
            if only not in name:
                continue
            _record(
                structure=True,
                module=name,
                cls=type(mod).__name__,
                children=[n for n, _ in mod.named_children()],
                params=[
                    {
                        "name": pn,
                        "shape": list(p.shape),
                        "requires_grad": p.requires_grad,
                        "placements": (
                            [str(x) for x in p.placements]
                            if isinstance(p, DTensor)
                            else "plain"
                        ),
                    }
                    for pn, p in mod.named_parameters(recurse=False)
                ],
            )


def param_names(model_parts) -> set[str]:
    return {
        n
        for part in model_parts
        for n, _ in part.named_parameters(remove_duplicate=False)
    }


class Bisect:
    """Record the parameter set at successive boundaries of the first step.

    Parameters registered at __init__ are missing by the end of step 1. That can
    only happen if an attribute is rebound to a non-Parameter, so the question is
    purely "which boundary loses them" -- deterministic, unlike the numerics.
    """

    def __init__(self, model_parts) -> None:
        self.parts = model_parts
        self.prev = param_names(model_parts)
        self.done = False
        _record(bisect="after_init", count=len(self.prev))

    def at(self, label: str) -> None:
        if self.done:
            return
        now = param_names(self.parts)
        lost = sorted(self.prev - now)
        _record(bisect=label, count=len(now), lost_here=lost)
        self.prev = now


def install() -> None:
    import torchtitan.train as T

    open(OUT, "w").close()
    original_init = T.Trainer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        want = os.environ.get("LORA_PROBE_STRUCTURE")
        if want:
            dump_structure(self.model_parts, want)

        bisect = Bisect(self.model_parts) if os.environ.get("LORA_PROBE_BISECT") else None
        original_step = self.train_step
        original_fb = self.forward_backward_step

        def fb(*a, **k):
            if bisect:
                bisect.at("step_start_before_forward_backward")
            out = original_fb(*a, **k)
            if bisect:
                bisect.at("after_forward_backward")
            return out

        def step(*a, **k):
            out = original_step(*a, **k)
            if bisect:
                bisect.at("after_optimizer_step")
                bisect.done = True
            compare_replicated_grads(self.model_parts, self.step)
            return out

        self.forward_backward_step = fb
        self.train_step = step

    T.Trainer.__init__ = patched_init


if __name__ == "__main__":
    install()
    from torchtitan.train import main

    main()
