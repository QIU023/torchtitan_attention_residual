"""#4380 tower benchmark, logbook kit only: per-GPU peak memory and step time of the Kimi K3 vision tower (the Kimi-K3
flavor's tower config, random weights, bf16 parameters, fp32 buffers) on large images and video. Every rank is one CP
rank and holds the whole micro-batch, as in training. Main's tree encodes every image whole on every rank; the PR's tree
splits images of at least dynamic_cp_min_patches over sub-CP groups (MODE=whole hands it no sub-groups).

env: CASE (key of CASES), AC none|sac|full (the recipe uses sac), WARM (default 2), ITERS (default 5),
     OUT (jsonl, rank 0 appends one line), TAG, MODE split|whole.
run: torchrun --nproc_per_node=<CP degree> tower_bench.py, from the tree's root.
"""

import json
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

from torchtitan.distributed import ParallelismContext
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.models.kimi_k3 import build_model_config

try:
    from torchtitan.models.kimi_k3.vision_cp import build_cp_subgroups
    from torchtitan.models.kimi_k3.vision_cp.plan import plan_dynamic_cp
except ImportError:  # main's tree
    build_cp_subgroups = plan_dynamic_cp = None

# [t, h, w] patch grids (patch 14, merge 2x2): 1008 px = 72 x 72 patches; video = 16 frames of 448 px in K3's 4-frame items.
CASES = {
    "448px": [[1, 32, 32]],
    "1008px": [[1, 72, 72]],
    "1456px": [[1, 104, 104]],
    "2016px": [[1, 144, 144]],
    "4032px": [[1, 288, 288]],
    "video16f448": [[4, 32, 32]] * 4,
    "4x2016px": [[1, 144, 144]] * 4,
}
GIB = 1024**3


def main() -> None:
    case, ac, mode = os.environ["CASE"], os.environ.get("AC", "sac"), os.environ.get("MODE", "split")
    warm, iters = int(os.environ.get("WARM", "2")), int(os.environ.get("ITERS", "5"))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda")

    pc = ParallelismContext(
        dp_replicate=1, dp_shard=1, cp=world, tp=1, pp=1, ep=1, world_size=world, enable_sequence_parallel=False
    )
    pc.build_mesh()

    config = build_model_config("Kimi-K3").vision_encoder
    torch.manual_seed(0)
    tower = config.build().to(device=device)
    tower.init_states()
    for p in tower.parameters():
        p.data = p.data.to(torch.bfloat16)
    if ac == "sac":
        SelectiveAC.Config().build().apply(tower)
    elif ac == "full":
        FullAC.Config().build().apply(tower)
    plan = None
    if build_cp_subgroups is not None and world > 1:
        subgroups = build_cp_subgroups(pc.get_mesh("cp"))
        if mode == "split":
            tower.set_cp_subgroups(subgroups)
            plan = plan_dynamic_cp(
                CASES[case], cp_size=world, kh=tower.merge_kernel_size[0], min_patches=tower.dynamic_cp_min_patches
            )

    grids = CASES[case]
    kh, kw = tower.merge_kernel_size
    num_patches = sum(t * h * w for t, h, w in grids)
    num_merged = sum((h // kh) * (w // kw) for _, h, w in grids)
    generator = torch.Generator(device=device).manual_seed(1)
    pixels = torch.randn(num_patches, tower.patch_embed.in_features, device=device, generator=generator).to(
        torch.bfloat16
    )
    upstream = torch.randn(
        num_merged, tower.projector.linear_2.out_features, device=device, generator=generator
    ).to(torch.bfloat16)
    grid_thw = torch.tensor(grids, device=device)

    def step() -> None:
        tower.zero_grad(set_to_none=True)
        with pc.activate_spmd():
            out = tower(pixels, grid_thw=grid_thw)
            (out * upstream).sum().backward()

    record = {
        "tag": os.environ.get("TAG", ""),
        "tree": os.path.basename(os.getcwd()),
        "case": case,
        "grids": grids,
        "patches": num_patches,
        "ac": ac,
        "cp": world,
        "mode": mode if plan is not None else "whole",
        "plan": None if plan is None else [plan.num_subgroups, plan.subgroup_size],
        "torch": torch.__version__,
    }
    status = "ok"
    try:
        for _ in range(warm):
            step()
        torch.cuda.synchronize()
        static = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(iters):
            dist.barrier()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        peak, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    except torch.OutOfMemoryError as e:
        status = "oom"
        sys.stderr.write(f"rank {rank} OOM: {str(e)[:300]}\n")
        static = peak = reserved = torch.cuda.max_memory_allocated()
        times = [float("nan")]

    mine = {"status": status, "static": static, "peak": peak, "reserved": reserved, "times": times}
    everyone = [None] * world
    dist.all_gather_object(everyone, mine)
    if rank == 0:
        statuses = {e["status"] for e in everyone}
        record["status"] = "ok" if statuses == {"ok"} else "oom"
        record["peak_gib"] = [round(e["peak"] / GIB, 3) for e in everyone]
        record["max_peak_gib"] = max(record["peak_gib"])
        record["max_reserved_gib"] = round(max(e["reserved"] for e in everyone) / GIB, 3)
        record["static_gib"] = round(max(e["static"] for e in everyone) / GIB, 3)
        per_iter = [max(e["times"][i] for e in everyone) for i in range(len(everyone[0]["times"]))]
        record["ms"] = round(1000 * statistics.median(per_iter), 1)
        record["ms_all"] = [round(1000 * t, 1) for t in per_iter]
        line = json.dumps(record)
        print("TOWER_BENCH " + line, flush=True)
        if os.environ.get("OUT"):
            with open(os.environ["OUT"], "a") as f:
                f.write(line + "\n")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
