"""Run the DEP runtime itself (3 gloo ranks, 12 stages, Interleaved1F1B, 6 micro-batches,
every micro-batch with one image) and record, per rank, the order in which the text
actions and the vision work actually execute during one training step (local only).

usage: python fig11_runtime.py <worktree> <cost_ratio>
"""

import json
import os
import sys
import tempfile

import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 3


def worker(rank, tree, ratio, store, out):
    sys.path.insert(0, os.path.join(tree, "tests", "unit_tests", "cpu"))
    sys.path.insert(0, tree)
    import test_kimi_k3_vision_dep as T
    from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.runtime import VisionDep
    from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.stage import (
        VisionDepPipelineStage,
    )

    T.NUM_STAGES, T.MICROBATCHES, T.STEPS = 12, 6, 1
    T.GRIDS = {mb: [(1, 1, 2)] for mb in range(6)}
    events = []

    def wrap_dep(name, tag):
        base = getattr(VisionDep, name)

        def f(self, mb, *a, **k):
            if self.active:
                events.append((tag, mb + 1))
            return base(self, mb, *a, **k)

        setattr(VisionDep, name, f)

    wrap_dep("_encode", "E")
    wrap_dep("_backward", "BW")
    fwd, bwd = VisionDepPipelineStage.forward_one_chunk, VisionDepPipelineStage.backward_one_chunk

    def f(self, chunk, *a, **k):
        if self._dep is not None and self._dep.active:
            events.append(("F", self.stage_index, int(chunk) + 1))
        return fwd(self, chunk, *a, **k)

    def b(self, chunk, *a, **k):
        if self._dep is not None and self._dep.active and self.has_backward:
            events.append(("B", self.stage_index, int(chunk) + 1))
        return bwd(self, chunk, *a, **k)

    VisionDepPipelineStage.forward_one_chunk, VisionDepPipelineStage.backward_one_chunk = f, b
    dist.init_process_group("gloo", init_method=f"file:///{store}", rank=rank, world_size=WORLD)

    class Run(T._VisionDepChecks):
        device_type = "cpu"

    run = Run()
    run.rank, run.world_size = rank, WORLD
    history, plan = run._run_pipeline(bubble=True, frozen_tower=False, evals=[], cost_ratio=ratio)
    import torch

    reference = T._run_single_device(False, torch.device("cpu"), False, 1.0)
    grads, losses = history[0]
    ref_grads, ref_losses = reference[0]
    same = sum(
        1
        for name, g in grads.items()
        if (g is None and ref_grads[name] is None)
        or (g is not None and ref_grads[name] is not None and torch.equal(g, ref_grads[name]))
    )
    loss_same = all(torch.equal(a, b) for a, b in zip(losses, ref_losses)) if losses else None
    with open(f"{out}.{rank}", "w") as fh:
        json.dump({"events": events, "grads_equal": [same, len(grads)], "losses_equal": loss_same}, fh)
    dist.destroy_process_group()


def summarize(events):
    text = [i for i, e in enumerate(events) if e[0] in ("F", "B")]
    first, last = text[0], text[-1]
    parts = {"before first text action": [], "between text actions": [], "after last text action": []}
    for i, e in enumerate(events):
        if e[0] in ("E", "BW"):
            where = ("before first text action" if i < first
                     else "after last text action" if i > last else "between text actions")
            parts[where].append(f"{'ViT fwd' if e[0] == 'E' else 'ViT bwd'} {e[1]}")
    return parts


if __name__ == "__main__":
    tree, ratio = sys.argv[1], float(sys.argv[2])
    d = tempfile.mkdtemp()
    mp.spawn(worker, args=(tree, ratio, os.path.join(d, "store"), os.path.join(d, "res")), nprocs=WORLD, join=True)
    for r in range(WORLD):
        res = json.load(open(os.path.join(d, f"res.{r}")))
        print(f"PP{r}: ", {k: v for k, v in summarize(res["events"]).items()})
        print("       step-1 grads bitwise equal to one device:", res["grads_equal"], "losses:", res["losses_equal"])
        seq = " ".join(
            (f"[{'E' if e[0] == 'E' else 'BW'}{e[1]}]" if e[0] in ("E", "BW") else f"{e[0]}{e[1]}.{e[2]}")
            for e in res["events"]
        )
        print("      ", seq)
