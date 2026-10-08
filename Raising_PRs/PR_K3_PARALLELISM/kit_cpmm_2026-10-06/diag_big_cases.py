"""#4380 larger split cases, logbook kit only: the debug tower in fp32 on every CP rank, split vs whole, forward and
per-parameter gradients, for large images and multi-frame video (the GPU test's grids stop at 240 patches).
Each case keeps a 144-patch image whole, as in the GPU test. env: CASES (comma separated keys), MINP (default 256).
run: torchrun --nproc_per_node=<CP degree> diag_big_cases.py, from the PR tree's root.
"""

import os
from dataclasses import replace

import torch
import torch.distributed as dist

from torchtitan.distributed import ParallelismContext
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.models.kimi_k3.vision_cp import build_cp_subgroups
from torchtitan.models.kimi_k3.vision_cp.plan import plan_dynamic_cp

CASES = {
    "2016px": [[1, 12, 12], [1, 144, 144]],
    "4032px": [[1, 12, 12], [1, 288, 288]],
    "short_last_rank": [[1, 12, 12], [1, 146, 144]],
    "video_4f_1008px": [[1, 12, 12], [4, 72, 72]],
    "video_16f_448px": [[1, 12, 12]] + [[4, 32, 32]] * 4,
    "mixed": [[1, 12, 12], [1, 144, 144], [4, 32, 32], [1, 72, 72]],
}


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ctx = ParallelismContext(
        dp_replicate=1, dp_shard=1, cp=world, tp=1, pp=1, ep=1, world_size=world, enable_sequence_parallel=False
    )
    ctx.build_mesh()
    min_patches = int(os.environ.get("MINP", "256"))
    config = replace(build_model_config("debugmodel").vision_encoder, dynamic_cp_min_patches=min_patches)
    torch.manual_seed(0)
    tower = config.build().to("cuda")
    tower.init_states()
    subgroups = build_cp_subgroups(ctx.get_mesh("cp"))
    kh, kw = tower.merge_kernel_size
    for name in os.environ.get("CASES", ",".join(CASES)).split(","):
        grids = CASES[name]
        plan = plan_dynamic_cp(grids, cp_size=world, kh=kh, min_patches=min_patches)
        generator = torch.Generator(device="cuda").manual_seed(1)
        n = sum(t * h * w for t, h, w in grids)
        m = sum((h // kh) * (w // kw) for _, h, w in grids)
        pixels = torch.randn(n, tower.patch_embed.in_features, device="cuda", generator=generator)
        upstream = torch.randn(m, tower.projector.linear_2.out_features, device="cuda", generator=generator)
        grid_thw = torch.tensor(grids, device="cuda")
        results = []
        for split in (False, True):
            tower.zero_grad(set_to_none=True)
            tower.set_cp_subgroups(subgroups if split else {})
            with ctx.activate_spmd():
                out = tower(pixels, grid_thw=grid_thw)
            (out * upstream).sum().backward()
            grads = {k: p.grad.clone() for k, p in tower.named_parameters()}
            if split:
                for v in grads.values():
                    dist.all_reduce(v)
                    v.div_(world)
            results.append((out.detach(), grads))
        (ref, ref_grads), (out, grads) = results
        fwd = ((out - ref).abs().max() / ref.abs().max()).item()
        rels = {k: ((grads[k] - ref_grads[k]).norm() / ref_grads[k].norm().clamp_min(1e-30)).item() for k in ref_grads}
        ratios = [(grads[k].norm() / ref_grads[k].norm().clamp_min(1e-30)).item() for k in ref_grads]
        worst = max(rels, key=rels.get)
        if rank == 0:
            print(
                f"BIG_CASE cp{world} {name} patches {n} plan "
                f"{None if plan is None else (plan.whole, plan.split, plan.num_subgroups, plan.subgroup_size)} "
                f"fwd_max_rel {fwd:.3e} grad_worst_rel {rels[worst]:.3e} ({worst}) "
                f"norm_ratio [{min(ratios):.6f}, {max(ratios):.6f}] "
                f"pass {fwd < 1e-5 and rels[worst] < 1e-4}",
                flush=True,
            )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
