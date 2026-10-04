"""ASCII timeline of a VisionDepPlan: one row per rank, one column per slot-time unit.
f/b = text forward/backward, '.' idle, digits = vision work of that micro-batch (1-based) placed in idle slots,
the prologue before '|' and the epilogue after '|'."""

import sys

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, sys.argv[1] + "/tests/unit_tests/cpu")
from test_kimi_k3_vision_dep_plan import _interleaved_order  # noqa: E402

from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan import (  # noqa: E402
    VisionDepPlan, _idle_runs, anchor_of,
)

pp, vp, m, ratio = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])
order = _interleaved_order(pp, vp, m)
plan = VisionDepPlan({mb: 1 for mb in range(m)}, num_microbatches=m, num_ranks=pp, stage0_rank=0,
                     trainable=True, pipeline_order=order, cost_ratio=ratio)
runs, step_end, times = _idle_runs(order)
width = int(round(times[step_end]))
for r in range(pp):
    row = ["."] * width
    for slot, action in enumerate(order[r][:step_end]):
        if action is None:
            continue
        kind = anchor_of(action)[0]
        for t in range(int(times[slot]), int(times[slot + 1])):
            row[t] = "f" if kind == "F" else "b"
    for (kind, mb), (rank, start, end) in plan.placed.items():
        if rank == r:
            for t in range(int(start), max(int(start) + 1, int(round(end)))):
                row[t] = str(mb + 1) if kind == "encode" else chr(ord("A") + mb)
    pro = "".join(str(mb + 1) for mb in plan.prologue[r]).rjust(2)
    epi = "".join(chr(ord("A") + mb) for mb in plan.epilogue[r]).ljust(2)
    print(f"PP{r} {pro}|{''.join(row)}|{epi}")
print("digits: vision forward of micro-batch n; letters A..: vision backward of micro-batch 1, 2, ...")
