"""Print each rank's allocator peaks at exit: max_memory_allocated is what a
parked-and-freed tensor lowers; max_memory_reserved (what the console and
nvidia-smi show) only stops growing. Enabled by being on PYTHONPATH."""

import atexit
import os


def _report():
    try:
        import torch

        if not torch.cuda.is_available() or not torch.cuda.is_initialized():
            return
        rank = os.environ.get("RANK", "?")
        alloc = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30
        print(f"[MEMPEAK] rank {rank} max_allocated={alloc:.2f}GiB max_reserved={reserved:.2f}GiB", flush=True)
    except Exception as exc:  # never let the report break a run
        print(f"[MEMPEAK] report failed: {exc}", flush=True)


atexit.register(_report)
