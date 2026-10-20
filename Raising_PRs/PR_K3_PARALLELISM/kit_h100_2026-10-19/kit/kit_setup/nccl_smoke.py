import os

import torch
import torch.distributed as dist

dist.init_process_group("nccl")
r = dist.get_rank()
torch.cuda.set_device(r)
x = torch.full((1 << 20,), float(r + 1), device="cuda")
dist.all_reduce(x)
if r == 0:
    print("nccl", torch.cuda.nccl.version(), "allreduce", float(x[0]), "expect", sum(range(1, dist.get_world_size() + 1)), flush=True)
dist.destroy_process_group()
