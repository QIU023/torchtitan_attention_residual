"""Per-axis replicated-gradient identity check: the matrix's floor-free assertion.

The parallelism matrix compares LOSSES across cells, and that has a floor. Two
things set it, both measured:

* Cross-layout per-parameter ratios have a floor of their own. Changing only the
  gradient-accumulation order on ONE GPU with no parallelism at all -- local batch
  2 against 4 -- gives A_log a median deviation of 0.267 and a norm-weighted max of
  0.0975. Parallel cells give 0.30-0.36 and 0.11-0.23. So a cross-layout number
  near 0.1 is the instrument, not a finding.
* The loss band is 0.0146 across eighteen cells, and the LoRA ``o_proj`` defect
  (fixed 2026-08-07) was worth 2.9e-02 in step-1 loss at tp2 -- inside a band the
  matrix already tolerates. A real defect can hide there.

This check has NO floor. A gradient whose placement on some mesh axis is
``Replicate`` must be bit-identical across that axis's ranks: the answer is exactly
0.0 or there is a bug. It needs no reference run, so it adds one assertion per cell
rather than a comparison between cells.

Three things it does that the earlier LoRA-era probe did not, each because getting
it wrong has already cost this project a conclusion:

1. **Per-axis subgroups, not the world group.** A gradient can be Replicate on the
   TP axis and Shard on the FSDP axis; gathering over the world then compares
   different shards and reports a false disagreement. Each axis is compared inside
   its own process group, where every peer holds the same shard of every other
   axis.
2. **Agreements are recorded, with their magnitude.** Recording only disagreements
   makes absence mean both "compared and agreed" and "never compared". It also
   separates real agreement from zero-agrees-with-zero: a zero-initialized
   parameter agrees trivially at step 1, so ``testable`` counts only agreements
   whose magnitude is non-zero. Run at least 3 steps.
3. **Skips are recorded with a reason.** A parameter that is a plain tensor, or has
   no gradient, is not evidence of anything and must not be counted as clean.

Exit code is 1 on any disagreement, so a matrix runner can treat it as pass/fail.

    torchrun --nproc_per_node=8 replicate_axis_check.py --module kimi_k3 \
      --config kimi_k3_debugmodel_report_arch --training.steps 3 ... \
      --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2
"""

from __future__ import annotations

import json
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

RANK = int(os.environ.get("RANK", "0"))
_BASE = os.environ.get("REPLICATE_CHECK_OUT", "/tmp/replicate_axis.jsonl")
OUT = _BASE.replace(".jsonl", f"_r{RANK}.jsonl")
_DISAGREEMENTS = [0]
_COMPARISONS = [0]
_TESTABLE = [0]
# World-reduced totals, refreshed at the end of every check() while the process
# group is still alive. The verdict reads these, never the per-rank counters.
_GLOBAL = [0, 0, 0]


def _record(**kw) -> None:
    with open(OUT, "a") as f:
        f.write(json.dumps(kw) + "\n")


def _replicate_axes(g: DTensor) -> list[str]:
    """Names of the mesh axes on which this gradient is Replicate."""
    names = g.device_mesh.mesh_dim_names
    if not names:
        return []
    return [
        name
        for name, placement in zip(names, g.placements)
        if isinstance(placement, Replicate)
    ]


def _reduce_totals() -> None:
    """World-sum the cumulative counters. Called from check(), where every rank
    participates, so the collective is balanced -- and while the process group is
    still alive, which it is not by the time main() returns."""
    if not dist.is_initialized():
        _GLOBAL[:] = [_COMPARISONS[0], _DISAGREEMENTS[0], _TESTABLE[0]]
        return
    # On the device the default group's backend owns: a CPU tensor on a NCCL group
    # raises "No backend type associated with device type cpu".
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t = torch.tensor(
        [_COMPARISONS[0], _DISAGREEMENTS[0], _TESTABLE[0]],
        dtype=torch.float64,
        device=dev,
    )
    dist.all_reduce(t)
    t = t.cpu()
    _GLOBAL[:] = [int(x) for x in t]


def _compare(step, name, axis, grp, tensor, placements) -> None:
    local = tensor.float().contiguous()
    if local.numel() == 0:
        # An empty local shard: this rank holds nothing of this parameter, which
        # happens at higher degrees on the axes this one is sharded along. Every
        # peer inside a Replicate axis's group holds the same shape, so either all
        # are empty or none are. Comparing nothing asserts nothing, so record it
        # rather than counting it -- and never let it reach .max(), which raises on
        # a zero-element tensor.
        _record(step=step, param=name, axis=axis, skipped="empty_local_shard")
        return
    size = dist.get_world_size(grp)
    buf = [torch.empty_like(local) for _ in range(size)]
    dist.all_gather(buf, local, group=grp)
    delta = max((o - buf[0]).abs().max().item() for o in buf[1:])
    mag = buf[0].abs().max().item()
    agrees = delta == 0.0
    _COMPARISONS[0] += 1
    if not agrees:
        _DISAGREEMENTS[0] += 1
    elif mag > 0:
        # An agreement at magnitude zero is not evidence: a zero-initialized
        # parameter agrees with itself trivially at step 1. Only non-zero
        # agreements are testable.
        _TESTABLE[0] += 1
    _record(
        step=step,
        param=name,
        axis=axis,
        axis_degree=size,
        agrees=agrees,
        max_delta=delta,
        ref_absmax=mag,
        placements=placements if placements is not None else "plain",
    )


def check(model_parts, step: int, parallel_dims=None) -> None:
    # Plain-tensor gradients carry no placement, so nothing about them can be read
    # off a DTensor. The general CP invariant -- CP shards the SEQUENCE, not the
    # parameters, so every CP rank must hold the same gradient after reduction --
    # does NOT hold in this stack: the "fsdp" mesh here is dp_shard x cp, so
    # parameters are FSDP-sharded across the cp axis too and their gradients come
    # back Shard, not Replicate. Measured: under cp2 and cp4 every one of 1182
    # records is "no_replicate_axis". The branch below is kept for a mesh where cp
    # is not folded in, and it records when it does not apply rather than assuming
    # either way.
    cp_group = None
    if parallel_dims is not None:
        cp = getattr(parallel_dims, "cp", 1)
        dp_shard = getattr(parallel_dims, "dp_shard", 1)
        tp = getattr(parallel_dims, "tp", 1)
        if cp > 1 and dp_shard == 1 and tp == 1:
            try:
                cp_group = parallel_dims.get_mesh("cp").get_group()
            except Exception as exc:
                # Record why, do not swallow. A silently absent group turns
                # "could not check" into "nothing to check", which is the
                # ambiguity this whole file exists to remove.
                _record(step=step, skipped="cp_group_unavailable", reason=repr(exc))
        else:
            _record(
                step=step,
                skipped="cp_plain_check_not_applicable",
                cp=cp,
                dp_shard=dp_shard,
                tp=tp,
            )

    for part in model_parts:
        for name, p in part.named_parameters(remove_duplicate=False):
            g = p.grad
            if g is None:
                _record(step=step, param=name, skipped="no_grad")
                continue
            if not isinstance(g, DTensor):
                if cp_group is None:
                    # No placement and no structural reason to expect equality.
                    # Saying so is the point: counting it as clean is how silence
                    # becomes a data point.
                    _record(step=step, param=name, skipped="plain_tensor")
                    continue
                _compare(step, name, "cp(plain)", cp_group, g.detach(), None)
                continue
            axes = _replicate_axes(g)
            if not axes:
                _record(
                    step=step,
                    param=name,
                    skipped="no_replicate_axis",
                    placements=[str(x) for x in g.placements],
                )
                continue
            local = g.to_local().detach()
            for axis in axes:
                grp = g.device_mesh[axis].get_group()
                if dist.get_world_size(grp) < 2:
                    _record(step=step, param=name, axis=axis, skipped="axis_degree_1")
                    continue
                _compare(
                    step, name, axis, grp, local, [str(x) for x in g.placements]
                )



def _finish_step() -> None:
    _reduce_totals()


def main() -> None:
    import torchtitan.train as T

    open(OUT, "w").close()
    original_init = T.Trainer.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        original_step = self.train_step

        def step(*a, **k):
            out = original_step(*a, **k)
            check(self.model_parts, self.step, self.parallel_dims)
            _finish_step()
            return out

        self.train_step = step

    T.Trainer.__init__ = patched_init

    from torchtitan.train import main as titan_main

    titan_main()

    # The verdict uses the WORLD-reduced counters gathered inside check(), not this
    # rank's own. Reducing here instead does not work: torchtitan destroys the
    # process group before main() returns, so a collective at this point is a
    # silent no-op -- measured, it printed PASS on a run where ranks 2 and 3 each
    # held 12 disagreements.
    comparisons, disagreements, testable = _GLOBAL

    # One verdict, computed once. Exit status alone cannot carry it: an import
    # error, an OOM and a real disagreement all exit 1, so a runner keying on the
    # code reads a crash as a finding. Absence of this line means the check never
    # ran, which is not the same as passing -- the collector must REQUIRE it.
    if disagreements:
        verdict = f"FAIL: {disagreements} replicated gradient(s) differ"
    elif comparisons == 0:
        # Every parameter was plain, gradient-less, or on a degree-1 axis. A cell
        # in that state has asserted nothing, and calling it a pass is how a
        # matrix leg looks verified without being measured.
        # NOASSERT, not FAIL: this cell is outside the check's coverage rather
        # than broken by it. A collector must be able to tell those apart, or
        # every CP-only and FSDP-only cell reads as a regression. Distinct exit
        # code for the same reason.
        verdict = "NOASSERT: no comparable gradients; this cell is not covered"
    elif testable == 0:
        # Everything agreed at magnitude zero. Zero agrees with zero; run more
        # steps or start warm.
        verdict = "FAIL: every agreement had magnitude zero; nothing was testable"
    else:
        verdict = "PASS"
    if RANK == 0:
        print(
            f"REPLICATE-CHECK {verdict} | comparisons={comparisons} "
            f"disagreements={disagreements} testable={testable}",
            flush=True,
        )
    if verdict.startswith("NOASSERT"):
        sys.exit(2)
    if verdict != "PASS":
        sys.exit(1)


if __name__ == "__main__":
    main()
