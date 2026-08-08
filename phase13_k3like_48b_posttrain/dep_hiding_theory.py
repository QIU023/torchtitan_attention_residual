# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Can the run-ahead hide the ViT, in theory? Hardware-independent.

Why this exists rather than a timing A/B. Measured at pp8 x vp4 on this box, all
three arms report mfu 0.12%, so the GPU compute in a 2063ms step is about 2.5ms and
everything else is PP communication, Python and data loading. The most that hiding
ALL vision compute could remove is therefore ~0.12% of the step, while the observed
arm-to-arm spread is 1.1-1.6% -- an order of magnitude larger. A latency A/B on this
hardware cannot resolve the effect in either direction, which is what the user said
at the outset about PCIe latency not extrapolating.

What CAN be decided without a clock: the Interleaved1F1B action schedule depends only
on (pp_size, num_stages, n_microbatches), so it can be built in one CPU process with
no model and no GPU.

The question is HOW MANY of the encodes can be hidden, not whether all of them can.
Asking the latter gives a useless answer: the report itself runs the first
micro-batches' ViT forwards synchronously upfront, so a model that demands every
encode be covered reports failure on a schedule that matches the report exactly (it
did, at vision forward #2, before this was corrected). What decides the count is
TIMING rather than capacity -- micro-batch m's encode must finish before m's forward
on the vision stage begins, so only bubbles EARLIER in the action order can pay for
it, and a bubble after the consumer is worthless however large the total.

Costs are expressed in units of one text-stage forward, so the only model-dependent
input is the ratio r = (one ViT forward) / (one text-stage forward). Measured from the
configs, r is 25.2 at report_arch_pp8vp4 and 0.057 at the real 2p8t_vl with one
1024-patch image -- ~440x apart, because the debug flavor's text side is tiny. Both
are reported rather than one, since a conclusion that holds only at r=0.057 is a
conclusion about the real model and must be labelled as such.

Usage:
    python3 dep_hiding_theory.py [--pp 8] [--vp 4] [--mb 32] [--r 25.2 --r 0.057]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class FakeStage:
    """Enough of ``_PipelineStageBase`` for the schedule's action generation.

    The schedule computes its action order from stage indices and counts alone; it
    only touches submodules when it RUNS. Building it this way keeps the analysis
    hardware-independent, which is the point -- a claim derived from a real 8-rank
    run would carry this box's timings with it.
    """

    stage_index: int
    num_stages: int
    group_rank: int
    is_first: bool = False
    is_last: bool = False

    def __post_init__(self) -> None:
        self.is_first = self.stage_index == 0
        self.is_last = self.stage_index == self.num_stages - 1


def build_order(pp_size: int, vp: int, n_microbatches: int):
    """Return ``{rank: [action|None]}`` for Interleaved1F1B, without a model."""
    from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B

    num_stages = pp_size * vp
    # Rank r owns stages r, r+pp, r+2pp, ... -- the interleaved (looped) assignment.
    stages_for_rank0 = [
        FakeStage(r, num_stages, 0) for r in range(0, num_stages, pp_size)
    ]
    sched = ScheduleInterleaved1F1B.__new__(ScheduleInterleaved1F1B)
    # Bypass __init__: it validates and wires real stages. Everything the action
    # generation reads is set explicitly here, so nothing is left implicit.
    sched._num_stages = num_stages
    sched.pp_group_size = pp_size
    sched._n_microbatches = n_microbatches
    sched.n_microbatches = n_microbatches
    sched._stages = stages_for_rank0
    sched.n_local_stages = len(stages_for_rank0)
    sched.stage_index_to_group_rank = {s: s % pp_size for s in range(num_stages)}
    sched.number_of_rounds = max(1, n_microbatches // pp_size)
    sched.microbatches_per_round = n_microbatches // sched.number_of_rounds
    if n_microbatches % sched.number_of_rounds != 0:
        raise ValueError(
            f"Interleaved1F1B needs n_microbatches ({n_microbatches}) to be a "
            f"multiple of the round count ({sched.number_of_rounds})"
        )
    order = {}
    for rank in range(pp_size):
        sched.rank = rank
        order[rank] = sched._calculate_single_rank_operations(rank)
    return order, num_stages


def analyse(order, vision_stage: int, r: float) -> dict:
    """How many encodes the bubbles on the vision rank can pay for, and how many
    must stay synchronous."""
    rank = None
    for rk, actions in order.items():
        if any(a is not None and a.stage_index == vision_stage for a in actions):
            rank = rk
            break
    if rank is None:
        raise ValueError(f"no rank holds stage {vision_stage}")

    actions = order[rank]
    bubbles = sum(1 for a in actions if a is None)

    # Prefix walk, one unit = one text-stage forward. A bubble BEFORE a micro-batch's
    # consumption point is idle time that encode can be moved into; a bubble after it
    # is useless, because the consumer has already run. So the budget accumulates as
    # the walk proceeds and is spent in order.
    #
    # The question is NOT "can every encode be hidden" -- the report says the first
    # micro-batches' ViT forwards run synchronously upfront, so some cannot, by its
    # own design. The question is HOW MANY can, which is what "most of the ViT
    # computation is hidden" quantifies. An encode that cannot be paid for stays
    # synchronous; it does not consume budget, it lengthens the step.
    budget = 0.0
    hidden = 0
    synchronous = 0
    first_hidden_at = None
    for idx, a in enumerate(actions):
        if a is None:
            budget += 1.0
            continue
        if a.stage_index != vision_stage or "FORWARD" not in str(a.computation_type):
            continue
        if budget >= r:
            budget -= r
            hidden += 1
            if first_hidden_at is None:
                first_hidden_at = idx
        else:
            synchronous += 1

    vision_forwards = hidden + synchronous
    return {
        "rank": rank,
        "slots": len(actions),
        "bubbles": bubbles,
        "vision_forwards": vision_forwards,
        "hidden": hidden,
        "synchronous": synchronous,
        "hidden_share": hidden / vision_forwards if vision_forwards else 0.0,
        "first_hidden_at": first_hidden_at,
        "bubble_share": ((hidden * r) / bubbles) if bubbles else float("inf"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp", type=int, default=8)
    ap.add_argument("--vp", type=int, default=4)
    ap.add_argument("--mb", type=int, default=32)
    ap.add_argument(
        "--r",
        type=float,
        action="append",
        help="ViT forward cost in units of one text-stage forward (repeatable)",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="sweep r over decades, so the conclusion does not rest on one estimate",
    )
    args = ap.parse_args()
    ratios = args.r or [25.2, 0.057]
    if args.sweep:
        ratios = [0.01, 0.057, 0.1, 0.3, 1.0, 3.0, 10.0, 25.2, 100.0]

    order, num_stages = build_order(args.pp, args.vp, args.mb)
    total_bubbles = sum(sum(1 for a in acts if a is None) for acts in order.values())
    print(
        f"pp={args.pp} vp={args.vp} stages={num_stages} microbatches={args.mb}: "
        f"{total_bubbles} bubbles across all ranks"
    )
    if args.sweep:
        print(
            f"\n{'r':>8}  {'hidden':>10}  {'sync':>5}  {'ViT cost':>9}  {'of step':>8}"
        )
        for r in ratios:
            res = analyse(order, vision_stage=0, r=r)
            vit_units = res["vision_forwards"] * r
            print(
                f"{r:>8.3f}  {res['hidden']:>3}/{res['vision_forwards']:<3} "
                f"{res['hidden_share'] * 100:>4.0f}%  {res['synchronous']:>5}  "
                f"{vit_units:>9.2f}  {vit_units / res['slots'] * 100:>7.1f}%"
            )
        print(
            "\n'ViT cost' is total vision compute in text-stage-forward units; "
            "'of step' is that\nagainst the vision rank's slot count -- the ceiling "
            "on any latency win from hiding it.\n"
            "\nThat last column assumes every slot costs the same wall time, i.e. that "
            "compute\ndominates the step. On this box it does NOT: pp8xvp4 reports mfu "
            "0.12%, so ~2.5ms\nof a 2063ms step is compute and the rest is PP comms, "
            "Python and data loading. Read\nthe column as the ceiling in a "
            "compute-bound deployment, not as a prediction here."
        )
        return

    for r in ratios:
        res = analyse(order, vision_stage=0, r=r)
        label = {25.2: "report_arch_pp8vp4 (debug)", 0.057: "2p8t_vl (real)"}.get(r, "")
        print(
            f"\nr={r} {label}\n"
            f"  vision rank {res['rank']}: {res['slots']} slots, "
            f"{res['bubbles']} bubbles\n"
            f"  {res['vision_forwards']} vision forwards -> "
            f"{res['hidden']} hidden in bubbles, "
            f"{res['synchronous']} stay synchronous "
            f"({res['hidden_share'] * 100:.1f}% hidden)\n"
            f"  bubble time consumed: {res['hidden'] * r:.2f} of "
            f"{res['bubbles']} units ({res['bubble_share'] * 100:.1f}%)\n"
            f"  first hideable encode at slot {res['first_hidden_at']}"
        )


if __name__ == "__main__":
    main()
