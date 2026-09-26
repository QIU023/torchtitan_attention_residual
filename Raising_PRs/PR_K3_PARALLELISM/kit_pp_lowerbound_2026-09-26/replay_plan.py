"""Replay a MemoryPlan from the inputs a probe run captured, next to that run's measured steps.

Usage: PYTHONPATH=<tree> python replay_plan.py <LB_ROOT> <cell> <rank> <traced step>

Prints, for the rank's compute actions: the profiled peak, the planned peak after the moves,
and the in-action peak measured in the traced step (steady state, so it carries the optimizer
state the profiled first step does not have).
"""

import json
import os
import pickle
import sys

from torch.distributed.pipelining.schedules import _Action

from torchtitan.models.kimi_k3.pipeline_parallel.activations import (
    compute_actions,
    MemoryPlan,
)


def main():
    root, cell, rank = sys.argv[1], sys.argv[2], int(sys.argv[3])
    folder = os.path.join(root, cell, "mem")
    with open(os.path.join(folder, f"plan_inputs{rank}.pkl"), "rb") as f:
        data = pickle.load(f)
    orders = {
        r: [None if a is None else _Action.from_str(a) for a in acts]
        for r, acts in data["orders"].items()
    }
    plan = MemoryPlan(orders, data["profiles"], **data["kwargs"])
    with open(os.path.join(folder, f"actions{rank}.json")) as f:
        measured = {a["action"]: a["peak_gib"] for a in json.load(f)["actions"]}
    profiled = data["profiles"][rank].peaks
    keys = compute_actions(orders[rank])
    gib = 2**30
    top = sorted(range(len(keys)), key=lambda i: -profiled[i])[:8]
    worst = sorted(range(len(keys)), key=lambda i: -measured.get(f"{keys[i][1]}{keys[i][0]}{keys[i][2]}", 0))[:8]
    print(f"rank {rank}: profiled max {max(profiled) / gib:.2f}, planned max {plan.peaks[rank] / gib:.2f}")
    print("| action | profiled | planned | measured (steady) |")
    print("|---|---:|---:|---:|")
    for i in sorted(set(top) | set(worst)):
        kind, stage, mb = keys[i]
        name = f"{stage}{kind}{mb}"
        print(
            f"| {name} | {profiled[i] / gib:.2f} | {plan.timelines[rank][i] / gib:.2f} | "
            f"{measured.get(name, float('nan')):.2f} |"
        )


if __name__ == "__main__":
    main()
