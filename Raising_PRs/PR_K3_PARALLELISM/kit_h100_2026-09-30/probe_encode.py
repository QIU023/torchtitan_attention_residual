"""Probe (09-30): does DEP's encode give the tower's features bitwise as DEP off computes them?

Run on one GPU in the DEP tree: torchrun --nproc_per_node=1 probe_encode.py  (env as run_dep_h100.sh, DEPV_TOWER=k3)

One K3 tower (bf16 parameters, seeded init) and one 1008 px image (5184 patches, seeded): the forward under no_grad as
DEP's encode runs it, the forward with grad as DEP's backward recomputes it, and the forward of an fp32 copy under
FSDP's bf16 mixed precision with grad as stage 0 runs it with DEP off; each twice, compared bitwise.
"""

import copy
import os
import sys

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

import dep_ratio_local as d


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(0)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(0)
    cfg = d._widened(enable_sp=False, seq_len=2048).vision_encoder
    with torch.device("meta"):
        tower = cfg.build()
    tower.to_empty(device="cuda")
    with torch.no_grad():
        tower.init_states()
    replica = copy.deepcopy(tower)
    with torch.no_grad():
        for p in replica.parameters():
            p.data = p.data.to(torch.bfloat16)
    per = 1008 // 14
    g = torch.Generator(device="cuda").manual_seed(1)
    pix = torch.randn(per * per, 3 * 14 * 14, device="cuda", generator=g).to(torch.bfloat16)
    grid = torch.tensor([[1, per, per]], device="cuda")

    def no_grad():
        with torch.no_grad():
            return replica(pix, grid_thw=grid)

    def with_grad():
        return replica(pix, grid_thw=grid).detach()

    fsdp = copy.deepcopy(tower)
    fully_shard(fsdp, mesh=init_device_mesh("cuda", (1,)),
                mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32))

    def fsdp_grad():
        return fsdp(pix, grid_thw=grid).detach()

    outs = {}
    for name, fn in (("no_grad", no_grad), ("grad", with_grad), ("fsdp_grad", fsdp_grad)):
        a, b = fn(), fn()
        print(f"{name}: repeat bitwise {torch.equal(a, b)}, shape {tuple(a.shape)} {a.dtype}")
        outs[name] = a
    for x, y in (("no_grad", "grad"), ("grad", "fsdp_grad"), ("no_grad", "fsdp_grad")):
        diff = (outs[x].float() - outs[y].float()).abs()
        print(f"{x} vs {y}: bitwise {torch.equal(outs[x], outs[y])}, max abs diff {diff.max().item():.3e}, "
              f"elements differing {int((diff > 0).sum())} of {diff.numel()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    sys.exit(main())
