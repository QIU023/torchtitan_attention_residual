"""Print where VisionDepPlan puts each micro-batch's encode and tower backward, per rank, in schedule slots."""

import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + "/tests/unit_tests/cpu")
from test_kimi_k3_vision_dep_plan import _interleaved_order  # noqa: E402

from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan import VisionDepPlan, _idle_runs  # noqa: E402


def show(pp, vp, m, ratio):
    order = _interleaved_order(pp, vp, m)
    plan = VisionDepPlan(
        {mb: 1 for mb in range(m)}, num_microbatches=m, num_ranks=pp, stage0_rank=0,
        trainable=True, pipeline_order=order, cost_ratio=ratio,
    )
    runs, step_end, times = _idle_runs(order)
    print(f"== pp {pp} x vp {vp}, {m} micro-batches, cost ratio {ratio}; step {times[step_end]:.0f} slot-time units")
    for r in range(pp):
        idle = " ".join(f"[{run.begin:.0f},{run.stop:.0f})" for run in runs[r])
        enc = sorted((w[1], s, e) for w, (rank, s, e) in plan.placed.items() if w[0] == "encode" and rank == r)
        bwd = sorted((w[1], s, e) for w, (rank, s, e) in plan.placed.items() if w[0] == "backward" and rank == r)
        fmt = lambda xs: " ".join(f"mb{mb + 1}@[{s:.1f},{e:.1f})" for mb, s, e in sorted(xs, key=lambda x: x[1]))
        print(f" rank {r}: idle {idle}")
        print(f"   encode upfront {[mb + 1 for mb in plan.prologue[r]]}  in idle: {fmt(enc)}")
        print(f"   backward in idle: {fmt(bwd)}  at the end {[mb + 1 for mb in plan.epilogue[r]]}")


for spec in sys.argv[2:]:
    pp, vp, m, ratio = spec.split(":")
    show(int(pp), int(vp), int(m), float(ratio))
