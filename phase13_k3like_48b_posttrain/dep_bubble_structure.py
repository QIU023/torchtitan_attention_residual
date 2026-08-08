"""Structural bubble analysis for DEP: slots, not seconds.

Wall-clock is not usable for this claim on this box. It is 8x RTX 5060 Ti over
PCIe, so the communication fraction of a step is nothing like the production
fabric the report describes, and any measured overlap would not extrapolate. So
this counts SCHEDULE SLOTS instead, which is a property of the pipeline schedule
and not of the hardware.

torch's own simulator does the work: ``_PipelineScheduleRuntime._simulate()`` dry-runs
the action list from every rank's perspective and emits, per rank, a list of actions
where **None is a bubble** -- a slot in which that rank could execute nothing because
a dependency was unmet. Its docstring is explicit that "the total number of simulator
steps can be used as a metric".

**The caveat that must travel with every number below.** The simulator's own
docstring: "The simulation is not high-fidelity and does not model overlapping of
compute and communication, or cuda streams." Every action costs one slot, so a ViT
forward counts the same as one text stage's forward. At real scale MoonViT-V2 is
447.4M against a text stage's share, so a slot is NOT a unit of work. What this
measures is whether the schedule PLACES vision work where the rank would otherwise
be idle -- an occupancy question, which is what "hidden within pipeline bubbles"
asserts structurally. It does not measure how much time that hides.

Run under torchrun; it dumps and exits before training, so it costs no step.

    KIMI_VIT_DEP=1 BUBBLE_OUT=/tmp/dep.json torchrun --nproc_per_node=8 \
      dep_bubble_structure.py --module kimi_k3 --config ... \
      --parallelism.pipeline_parallel_degree 8 \
      --parallelism.pipeline_parallel_schedule Interleaved1F1B ...
"""

from __future__ import annotations

import json
import os
import sys

import torch.distributed as dist


# Actions the dependency simulator does not model. They are FSDP parameter
# management, not pipeline dependencies, and leaving them in makes _simulate raise
# "Unsupported action type 0UNSHARD" instead of skipping them.
_NON_PIPELINE = ("UNSHARD", "RESHARD", "REDUCE_GRAD")


def _analyse(schedule, vision_stages: set[int]) -> dict:
    from torch.distributed.pipelining.schedules import _simulate_comms_compute

    order = {
        rank: [
            a
            for a in acts
            if a is None or not any(t in str(a.computation_type) for t in _NON_PIPELINE)
        ]
        for rank, acts in schedule.pipeline_order_with_comms.items()
    }
    sim = _simulate_comms_compute(
        order,
        lambda s: schedule.stage_index_to_group_rank[s],
        schedule._num_stages,
    )
    out = {"per_rank": {}, "vision_stages": sorted(vision_stages)}
    for rank, slots in sorted(sim.items()):
        total = len(slots)
        bubbles = sum(1 for a in slots if a is None)
        vision = sum(
            1 for a in slots if a is not None and a.stage_index in vision_stages
        )
        by_stage: dict[int, int] = {}
        for a in slots:
            if a is not None:
                by_stage[a.stage_index] = by_stage.get(a.stage_index, 0) + 1
        out["per_rank"][rank] = {
            "slots": total,
            "bubbles": bubbles,
            "vision_slots": vision,
            "stages_held": sorted(by_stage),
            "actions_per_stage": {str(k): v for k, v in sorted(by_stage.items())},
        }
    out["total_slots"] = max(v["slots"] for v in out["per_rank"].values())
    out["total_bubbles"] = sum(v["bubbles"] for v in out["per_rank"].values())
    return out


def main() -> None:
    import torchtitan.train as T

    original_init = T.Trainer.__init__

    def patched(self, *a, **k):
        original_init(self, *a, **k)
        sched = getattr(self, "pp_schedule", None)
        if sched is None:
            if dist.get_rank() == 0:
                print("BUBBLE no pp_schedule; PP is not enabled", flush=True)
            sys.exit(0)

        # Under DEP the vision stage is stage 0 (and 0..n-1 once n_vit > 1). It is
        # identified from the FQN plan rather than assumed, so a future layout change
        # cannot silently mislabel it.
        from torchtitan.models.kimi_k3.pipeline_adapter import (
            _DEP_VISION_FQN,
            dep_enabled,
        )

        vision: set[int] = set()
        if dep_enabled():
            fqns = k.get("parallelism", a[0] if a else None)
            plan = getattr(
                k.get("parallelism", None), "module_fqns_per_model_part", None
            )
            if plan:
                vision = {
                    i
                    for i, names in enumerate(plan)
                    if any(str(n).startswith(_DEP_VISION_FQN) for n in names)
                }
            del fqns
            if not vision:
                vision = {0}

        report = _analyse(sched, vision)
        if dist.get_rank() == 0:
            path = os.environ.get("BUBBLE_OUT")
            if path:
                with open(path, "w") as f:
                    json.dump(report, f, indent=2)
            print("BUBBLE " + json.dumps(report), flush=True)
        # Exit before training: the schedule is all this needs, and a real step at
        # pp8 x 16 microbatches would cost memory for nothing.
        sys.exit(0)

    T.Trainer.__init__ = patched
    from torchtitan.train import main as titan_main

    titan_main()


if __name__ == "__main__":
    main()
