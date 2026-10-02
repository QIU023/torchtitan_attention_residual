"""A/B of the DEP pipeline on two trees (local only): four gloo ranks run the CPU test's
pp4 x vpp2 Interleaved1F1B model with the decoupled encoder process, and every step's
losses, eval losses, gradients and the plan are saved; compare.py checks them bitwise.

usage: python ab_pipeline.py <tree> <out_dir>
"""

import os
import sys
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 4
CELLS = [
    # (name, bubble, frozen_tower, cost_ratio, gelu, lr)
    ("before_after", False, False, 0.5, False, 1.0),
    ("bubble", True, False, 0.5, False, 1.0),
    ("bubble_late_grad", True, False, 0.25, False, 1.0),
    ("bubble_frozen", True, True, 0.5, False, 1.0),
    ("bubble_gelu", True, False, 0.5, True, 1e-3),
    ("before_after_gelu", False, False, 0.25, True, 1e-3),
]
FIELDS = ("encode_rank", "backward_rank", "prologue", "epilogue", "anchored", "posts", "placed")


def worker(rank, tree, out_dir, store_dir):
    sys.path.insert(0, os.path.join(tree, "tests", "unit_tests", "cpu"))
    sys.path.insert(0, tree)
    import test_kimi_k3_vision_dep as T

    for name, bubble, frozen, ratio, gelu, lr in CELLS:
        store = os.path.join(store_dir, f"store_{name}")
        dist.init_process_group(
            "gloo", init_method=f"file:///{store}", rank=rank, world_size=WORLD
        )

        class Run(T._VisionDepChecks):
            device_type = "cpu"

        run = Run()
        run.rank, run.world_size, run.gelu, run.lr = rank, WORLD, gelu, lr
        evals = []
        history, plan = run._run_pipeline(
            bubble=bubble, frozen_tower=frozen, evals=evals, cost_ratio=ratio
        )
        torch.save(
            {
                "history": history,
                "evals": evals,
                "plan": {f: repr(getattr(plan, f)) for f in FIELDS},
            },
            os.path.join(out_dir, f"{name}.{rank}.pt"),
        )
        dist.destroy_process_group()


if __name__ == "__main__":
    tree, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    mp.spawn(worker, args=(tree, out_dir, tempfile.mkdtemp()), nprocs=WORLD, join=True)
    print("done", tree)
