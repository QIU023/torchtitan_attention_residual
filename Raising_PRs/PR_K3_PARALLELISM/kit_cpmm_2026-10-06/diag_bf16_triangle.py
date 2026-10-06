"""bf16 error of the split tower and of main's whole tower, each against an fp32 whole tower, over the CP ranks
(torchrun, world = CP size); logbook kit only. Env: GRIDS (t x h x w list), MINP (dynamic_cp_min_patches)."""
import copy
import os
from dataclasses import replace

import torch
import torch.distributed as dist

from torchtitan.distributed import ParallelismContext
from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.models.kimi_k3.vision_encoder import build_cp_subgroups

grids = [list(map(int, g.split("x"))) for g in os.environ.get("GRIDS", "1x12x16,1x16x24").split(",")]
rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(rank)
dist.init_process_group("nccl")
ctx = ParallelismContext(
    dp_replicate=1, dp_shard=1, cp=world, tp=1, pp=1, ep=1, world_size=world, enable_sequence_parallel=False
)
ctx.build_mesh()
config = replace(
    build_model_config("debugmodel").vision_encoder, dynamic_cp_min_patches=int(os.environ.get("MINP", "128"))
)
torch.manual_seed(0)
tower32 = config.build().to("cuda")
tower32.init_states()
tower16 = copy.deepcopy(tower32)
for param in tower16.parameters():  # parameters only, buffers stay fp32, as FSDP's param_dtype does
    param.data = param.data.to(torch.bfloat16)
subgroups = build_cp_subgroups(ctx.get_mesh("cp").get_group())
kh, kw = tower32.merge_kernel_size
g = torch.Generator(device="cuda").manual_seed(1)
n = sum(t * h * w for t, h, w in grids)
m = sum((h // kh) * (w // kw) for _, h, w in grids)
pixels = torch.randn(n, tower32.patch_embed.in_features, device="cuda", generator=g)
upstream = torch.randn(m, tower32.projector.linear_2.out_features, device="cuda", generator=g)
grid_thw = torch.tensor(grids, device="cuda")


def run(tower, dtype, split):
    tower.zero_grad(set_to_none=True)
    tower.set_cp_subgroups(subgroups if split else {})
    with ctx.activate_spmd():
        out = tower(pixels.to(dtype), grid_thw=grid_thw)
    (out.float() * upstream).sum().backward()
    grads = {k: p.grad.float().clone() for k, p in tower.named_parameters()}
    if split:
        for v in grads.values():
            dist.all_reduce(v)
            v.div_(world)
    return out.detach().float(), grads


ref, rg = run(tower32, torch.float32, False)
whole, wg = run(tower16, torch.bfloat16, False)
split, sg = run(tower16, torch.bfloat16, True)


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


if rank == 0:
    print(f"grids {grids} world {world} bank {tuple(ref.shape)}")
    print(f"forward rel norm: whole_bf16 vs fp32 {rel(whole, ref):.3e}  split_bf16 vs fp32 {rel(split, ref):.3e}"
          f"  split vs whole (bf16) {rel(split, whole):.3e}  bitwise {torch.equal(split, whole)}")
    ew = torch.tensor([rel(wg[k], rg[k]) for k in rg])
    es = torch.tensor([rel(sg[k], rg[k]) for k in rg])
    esw = torch.tensor([rel(sg[k], wg[k]) for k in rg])
    print(f"grads over {len(rg)} params, rel norm vs fp32: whole median {ew.median():.3e} max {ew.max():.3e};"
          f" split median {es.median():.3e} max {es.max():.3e}; split vs whole median {esw.median():.3e}"
          f" max {esw.max():.3e}; params where split error > 2x whole error: {(es > 2 * ew).sum().item()}")
dist.destroy_process_group()
