"""Diagnose split-vs-whole tower gradients on 4 ranks (cp4): per-parameter relative difference, logbook kit only."""
import os
import sys
from dataclasses import replace

import torch
import torch.distributed as dist

from torchtitan.distributed import ParallelismContext
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.models.kimi_k3.vision_encoder import build_cp_subgroups

grids = [list(map(int, g.split("x"))) for g in os.environ.get("GRIDS", "1x12x12,1x16x16").split(",")]
rank = int(os.environ["RANK"]); torch.cuda.set_device(rank); dist.init_process_group("nccl")
ctx = ParallelismContext(dp_replicate=1, dp_shard=1, cp=4, tp=1, pp=1, ep=1, world_size=4, enable_sequence_parallel=False)
ctx.build_mesh()
config = replace(build_model_config("debugmodel").vision_encoder, dynamic_cp_min_patches=200)
torch.manual_seed(0)
tower = config.build().to("cuda"); tower.init_states()
subgroups = build_cp_subgroups(ctx.get_mesh("cp").get_group())
kh, kw = tower.merge_kernel_size
g = torch.Generator(device="cuda").manual_seed(1)
n = sum(t * h * w for t, h, w in grids); m = sum((h // kh) * (w // kw) for _, h, w in grids)
pixels = torch.randn(n, tower.patch_embed.in_features, device="cuda", generator=g)
upstream = torch.randn(m, tower.projector.linear_2.out_features, device="cuda", generator=g)
grid_thw = torch.tensor(grids, device="cuda")
res = []
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
            v.div_(4)
    res.append((out.detach(), grads))
(ref, rg), (out, sg) = res
if rank == 0:
    print("forward max rel", ((out - ref).abs().max() / ref.abs().max()).item())
    worst = []
    for k in rg:
        d = ((sg[k] - rg[k]).norm() / rg[k].norm().clamp_min(1e-30)).item()
        ratio = (sg[k].norm() / rg[k].norm().clamp_min(1e-30)).item()
        worst.append((d, ratio, k))
    for d, ratio, k in sorted(worst, reverse=True)[:10]:
        print(f"  rel {d:.3e}  norm ratio {ratio:.6f}  {k}")
dist.destroy_process_group()
