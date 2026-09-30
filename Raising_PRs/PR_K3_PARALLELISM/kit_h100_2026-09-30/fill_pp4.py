"""Modelled DEP bubble placement on the H100 layout (pp4 x vpp4, Interleaved1F1B) against the encode cost ratio.

Usage: python fill_pp4.py <DEP tree>

The planner's own model (`plan_dep` with the schedule's action order): time in text-stage forwards, a forward 1, a
backward 2, a slot as long as its longest action; a micro-batch's encode costs `cost_ratio`, its recompute plus backward
three times that. Every micro-batch carries one image (the heaviest case of the campaign's k 13 level is 13 of 16).
Prints how many encodes and backwards the planner puts in idle slots and the share of idle time they fill, for the
campaign's measured ratio (3.61, one 1008 px encode over a 16 stage split's middle stage at seq 2048) and smaller ones
down to the K3 report's range (0.02 to 0.32, DEP_REPORTS_READING_2026-09-30.md).
"""

import importlib.util
import sys

W = sys.argv[1]
sys.path.insert(0, W)
spec = importlib.util.spec_from_file_location("t", f"{W}/tests/unit_tests/cpu/test_kimi_k3_dep_plan.py")
t = importlib.util.module_from_spec(spec)
sys.modules["t"] = t
spec.loader.exec_module(t)
from torchtitan.models.kimi_k3.pipeline_parallel import dep_plan  # noqa: E402


def run(pp, vp, m, ratio, carriers):
    order = t._interleaved_order(pp, vp, m)
    loads = {mb: 5184 for mb in range(m) if mb in carriers}
    plan = dep_plan.plan_dep(loads, num_microbatches=m, num_ranks=pp, stage0_rank=0, trainable=True,
                             pipeline_order=order, cost_ratio=ratio)
    runs, step_end, times = dep_plan._idle_runs(order)
    idle = sum(x.stop - x.begin for rr in runs.values() for x in rr)
    placed = sum(e - s for (_, (_, s, e)) in plan.placed.items())
    enc = sum(1 for k in plan.placed if k[0] == "encode")
    bwd = sum(1 for k in plan.placed if k[0] == "backward")
    return enc, bwd, len(loads), placed / idle if idle else 0.0, idle / (pp * times[step_end])


def main():
    print("| layout | M | micro-batches with an image | cost ratio | encodes in idle slots | backwards in idle slots | "
          "idle share of the step (all ranks) | idle filled |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for m in (8, 16):
        for per16 in (7, 16):
            carriers = {i for i in range(m) if (i + 1) * per16 // 16 > i * per16 // 16}
            for ratio in (3.61, 2.0, 1.0, 0.5, 0.3, 0.1):
                enc, bwd, n, fill, idle = run(4, 4, m, ratio, carriers)
                print(f"| pp4 x vpp4 | {m} | {n} | {ratio} | {enc} | {bwd} | {idle:.0%} | {fill:.0%} |")


if __name__ == "__main__":
    main()
