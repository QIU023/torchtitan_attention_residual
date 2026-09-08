"""Minimal repro: a pipeline stage whose forward starts with a cat on its input hands back a strided (non-dense)
input gradient; the previous stage sizes its gradient receive buffer with torch.empty_strided from that metadata,
and NCCL P2P rejects it: 'Tensors for P2P must be non-overlapping and dense'. Run: torchrun --nproc_per_node=2 pp_dense_repro.py"""
import os, torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe

class Stage0(nn.Module):
    def __init__(s, d): super().__init__(); s.lin = nn.Linear(d, d)
    def forward(s, x): return s.lin(x)

class Stage1(nn.Module):
    def __init__(s, d): super().__init__(); s.lin = nn.Linear(2 * d, 1)
    def forward(s, x):
        # cat with a constant: x's gradient is grad.narrow(-1, 0, d), strides (2d, 1) -> non-dense
        return s.lin(torch.cat([x, torch.ones_like(x)], dim=-1)).sum(-1)

def main():
    dist.init_process_group("nccl"); rank = dist.get_rank(); dev = torch.device("cuda", rank); torch.cuda.set_device(dev)
    d, B = 8, 4
    mod = (Stage0(d) if rank == 0 else Stage1(d)).to(dev)
    stage = PipelineStage(mod, rank, 2, dev)
    sched = ScheduleGPipe(stage, n_microbatches=2, loss_fn=lambda out, tgt: (out - tgt).pow(2).mean())
    x = torch.randn(B, d, device=dev); tgt = torch.randn(B, device=dev)
    try:
        if rank == 0: sched.step(x)
        else: sched.step(target=tgt)
        print(f"rank {rank}: OK (no error)", flush=True)
    except Exception as e:
        print(f"rank {rank}: {type(e).__name__}: {str(e)[:200]}", flush=True)
    dist.barrier(); dist.destroy_process_group()

if __name__ == "__main__": main()
