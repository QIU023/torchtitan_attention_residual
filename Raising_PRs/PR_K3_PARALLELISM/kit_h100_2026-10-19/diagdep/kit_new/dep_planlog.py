"""DEP diagnosis (logbook kit only): dep_new_local's bubble cell; rank 0 prints the inputs and the result of the first
VisionDepPlan: per-micro-batch patch loads, cost ratio, ranks, stage-0 rank, each rank's action order and the placement."""
import hashlib
import os

import torch.distributed as dist

import dep_new_local


def _patch() -> None:
    from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan import VisionDepPlan

    original = VisionDepPlan.__init__
    done = [False]

    def init(self, loads, **kwargs):
        original(self, loads, **kwargs)
        if done[0] or not dist.is_initialized() or dist.get_rank() != 0:
            return
        done[0] = True
        order = kwargs.get("pipeline_order") or {}
        print(f"PLANLOG loads {sorted(dict(loads).items())}", flush=True)
        print(f"PLANLOG cost_ratio {kwargs.get('cost_ratio')} num_microbatches {kwargs.get('num_microbatches')} "
              f"num_ranks {kwargs.get('num_ranks')} stage0_rank {kwargs.get('stage0_rank')}", flush=True)
        for rank in sorted(order):
            text = " ".join("-" if a is None else str(a) for a in order[rank])
            print(f"PLANLOG order rank {rank} len {len(order[rank])} sha {hashlib.sha1(text.encode()).hexdigest()[:12]}", flush=True)
            print(f"PLANLOG order rank {rank}: {text}", flush=True)
        print(f"PLANLOG prologue {self.prologue} epilogue {self.epilogue}", flush=True)
        print(f"PLANLOG placed {self.placed}", flush=True)

    VisionDepPlan.__init__ = init
    print(f"[dep_planlog] rank {os.environ.get('RANK')} VisionDepPlan patched", flush=True)


def w_dep_bubble():
    _patch()
    return dep_new_local.w_dep_bubble()
