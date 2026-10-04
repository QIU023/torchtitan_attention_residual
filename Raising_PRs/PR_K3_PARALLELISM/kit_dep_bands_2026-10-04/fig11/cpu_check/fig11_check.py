"""Independent check of VisionDepPlan against Figure 11 of the K3 report (local only).

The figure, read off the image (1-based micro-batches, PP0..PP2):
  ViT forward: before the schedule PP0 {1}, PP1 {2}, PP2 {3}; in the opening bubble PP1 {4}, PP2 {5, 6}
  ViT backward: in the closing bubble PP1 {1}, PP2 {2, 3}; after the schedule PP0 {4}, PP1 {5}, PP2 {6}
  no vision work in the steady state.

usage: python fig11_check.py <worktree>
"""

import sys

sys.path.insert(0, sys.argv[1])
from torch.distributed.pipelining.schedules import _Action, ScheduleInterleaved1F1B  # noqa: E402

from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan import (  # noqa: E402
    _idle_runs,
    VisionDepPlan,
)

FIGURE = {
    "prologue": {0: [1], 1: [2], 2: [3]},
    "opening": {1: [4], 2: [5, 6]},
    "closing": {1: [1], 2: [2, 3]},
    "epilogue": {0: [4], 1: [5], 2: [6]},
}


def interleaved_order(pp, vp, m):
    """torch's own ScheduleInterleaved1F1B action order, built without process groups."""
    n = pp * vp
    s = ScheduleInterleaved1F1B.__new__(ScheduleInterleaved1F1B)
    s._num_stages, s.pp_group_size, s._n_microbatches, s.n_microbatches = n, pp, m, m
    s.stage_index_to_group_rank = {i: i % pp for i in range(n)}
    s.number_of_rounds = max(1, m // pp)
    s.microbatches_per_round = m // s.number_of_rounds

    class _S:
        def __init__(self, i):
            self.stage_index, self.num_stages = i, n
            self.group_rank, self.is_first, self.is_last = i % pp, i == 0, i == n - 1

    order = {}
    for r in range(pp):
        s._stages = [_S(i) for i in range(r, n, pp)]
        s.n_local_stages, s.rank = len(s._stages), r
        order[r] = s._calculate_single_rank_operations(r)
    return order


def layout(plan, order):
    runs, step_end, _ = _idle_runs(order)
    out = {"prologue": {}, "opening": {}, "closing": {}, "epilogue": {}, "steady": {}}
    for r, mbs in plan.prologue.items():
        if mbs:
            out["prologue"][r] = [mb + 1 for mb in mbs]
    for r, mbs in plan.epilogue.items():
        if mbs:
            out["epilogue"][r] = [mb + 1 for mb in mbs]
    for (kind, mb), (r, start, end) in sorted(plan.placed.items(), key=lambda x: (x[1][0], x[1][1])):
        first_busy = min(i for i, a in enumerate(order[r]) if a is not None)
        last_busy = max(i for i, a in enumerate(order[r]) if a is not None)
        run = next(run for run in runs[r] if run.begin <= start < run.stop or run.begin <= start <= run.stop)
        if kind == "encode" and run.start == 0 and run.end == first_busy:
            where = "opening"
        elif kind == "backward" and run.start == last_busy + 1 and run.end == step_end:
            where = "closing"
        else:
            where = "steady"
        out[where].setdefault(r, []).append(mb + 1)
    return {k: v for k, v in out.items() if v or k != "steady"}


def show(order, plan, title):
    print(f"== {title}")
    lay = layout(plan, order)
    for k in ("prologue", "opening", "steady", "closing", "epilogue"):
        if k in lay:
            print(f"   {k:9s}", {r: lay[k][r] for r in sorted(lay[k])})
    return lay


def main():
    order = interleaved_order(3, 4, 6)
    for r, acts in order.items():
        print(f"PP{r}:", " ".join("." if a is None else f"{str(a.computation_type)[0]}{a.stage_index}m{a.microbatch_index + 1}" for a in acts))
    bad = 0
    for ratio in (0.01, 0.05, 0.1, 0.2, 0.3, 1 / 3, 0.34, 0.5, 1.0):
        plan = VisionDepPlan(
            {mb: 100 for mb in range(6)}, num_microbatches=6, num_ranks=3, stage0_rank=0,
            trainable=True, pipeline_order=order, cost_ratio=ratio,
        )
        lay = show(order, plan, f"pp3 x vp4, M6, uniform loads, cost ratio {ratio:.3g}")
        same = {k: lay.get(k, {}) for k in FIGURE} == FIGURE and not lay.get("steady")
        print("   matches Figure 11:", same)
        bad += not same and ratio <= 1 / 3
    # images on only some micro-batches, uneven patch counts
    for loads in ({0: 100, 1: 400, 2: 50, 3: 100, 4: 300, 5: 100}, {0: 100, 2: 100, 3: 100, 5: 100}):
        plan = VisionDepPlan(loads, num_microbatches=6, num_ranks=3, stage0_rank=0, trainable=True,
                             pipeline_order=order, cost_ratio=0.1)
        show(order, plan, f"pp3 x vp4, M6, loads {loads}, ratio 0.1")
    # deeper shapes: who takes what
    for pp, vp, m in ((4, 4, 8), (4, 4, 16), (8, 4, 32), (16, 2, 32)):
        order = interleaved_order(pp, vp, m)
        plan = VisionDepPlan({mb: 100 for mb in range(m)}, num_microbatches=m, num_ranks=pp, stage0_rank=0,
                             trainable=True, pipeline_order=order, cost_ratio=0.1)
        lay = show(order, plan, f"pp{pp} x vp{vp}, M{m}, ratio 0.1")
        counts = {k: {r: len(v) for r, v in sorted(lay.get(k, {}).items())} for k in ("opening", "closing")}
        print("   counts", counts, "steady:", lay.get("steady", {}))
    print("ratios <= 1/3 not matching the figure:", bad)


main()
