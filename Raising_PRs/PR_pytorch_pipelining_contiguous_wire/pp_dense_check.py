"""The repro with a numerics reference: the pipelined loss and the input-side gradient must equal a single-process run."""
import torch, torch.nn as nn, torch.distributed as dist
from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
torch.manual_seed(0)
d, B = 8, 4
class Stage0(nn.Module):
    def __init__(s): super().__init__(); s.lin = nn.Linear(d, d)
    def forward(s, x): return s.lin(x)
class Stage1(nn.Module):
    def __init__(s): super().__init__(); s.lin = nn.Linear(2 * d, 1)
    def forward(s, x): return s.lin(torch.cat([x, torch.ones_like(x)], dim=-1)).sum(-1)
def main():
    dist.init_process_group("nccl"); rank = dist.get_rank(); dev = torch.device("cuda", rank); torch.cuda.set_device(dev)
    s0, s1 = Stage0(), Stage1()  # same init on both ranks (seed 0)
    x = torch.randn(B, d); tgt = torch.randn(B)
    loss_fn = lambda out, t: (out - t).pow(2).mean()
    # reference on rank 0, single process
    if rank == 0:
        r0, r1 = Stage0(), Stage1(); r0.load_state_dict(s0.state_dict()); r1.load_state_dict(s1.state_dict())
        r0, r1 = r0.to(dev), r1.to(dev)
        mb = x.to(dev).chunk(2); tt = tgt.to(dev).chunk(2)
        for a, t in zip(mb, tt):
            (loss_fn(r1(r0(a)), t) / 2).backward()  # GPipe averages the loss over microbatches
        ref_grad = r0.lin.weight.grad.clone()
    mod = (s0 if rank == 0 else s1).to(dev)
    stage = PipelineStage(mod, rank, 2, dev)
    sched = ScheduleGPipe(stage, n_microbatches=2, loss_fn=loss_fn)
    if rank == 0: sched.step(x.to(dev))
    else: sched.step(target=tgt.to(dev))
    if rank == 0:
        pp_grad = mod.lin.weight.grad
        print(f"rank 0: stage-0 weight grad, pipeline vs single process: max|d| = {(pp_grad - ref_grad).abs().max().item():.3e} (bitwise: {torch.equal(pp_grad, ref_grad)})", flush=True)
    dist.barrier(); dist.destroy_process_group()
if __name__ == "__main__": main()
