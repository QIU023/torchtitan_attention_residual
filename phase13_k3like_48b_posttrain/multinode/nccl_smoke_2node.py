"""2-node NCCL smoke: rendezvous, a size-swept all-reduce, and busbw.

Run via launch_smoke.sh (which sets the env and torchrun args). Gates:
  1. init_process_group completes across 16 ranks (rendezvous + NCCL
     bootstrap both cross the node boundary),
  2. all-reduce results are CORRECT (sum of rank ids, checked exactly),
  3. prints per-size algbw/busbw so the overlay's real throughput is on
     record before any training run depends on it.

busbw for ring all-reduce = algbw * 2*(n-1)/n; the cross-node links carry
~1/n of ring traffic per direction, so busbw ~= the bottleneck link speed
when the LAN is the bottleneck.
"""

import os
import time

import torch
import torch.distributed as dist


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    if rank == 0:
        print(f"[SMOKE] world={world} nccl bootstrap OK", flush=True)

    # correctness first: exact integer all-reduce
    t = torch.full((1024,), float(rank), device="cuda")
    dist.all_reduce(t)
    expect = world * (world - 1) / 2.0
    ok = bool((t == expect).all().item())
    if not ok:
        raise SystemExit(f"[SMOKE] rank {rank}: WRONG all-reduce result")
    if rank == 0:
        print(f"[SMOKE] correctness: all-reduce sum == {expect:.0f} on every element", flush=True)

    # bandwidth sweep
    for mib in (1, 8, 64, 256):
        x = torch.randn(mib * 1024 * 1024 // 4, device="cuda")
        for _ in range(3):  # warmup
            dist.all_reduce(x)
        torch.cuda.synchronize()
        n_iter = 10
        t0 = time.time()
        for _ in range(n_iter):
            dist.all_reduce(x)
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n_iter
        algbw = mib / 1024 / dt  # GiB/s
        busbw = algbw * 2 * (world - 1) / world
        if rank == 0:
            print(
                f"[SMOKE] all-reduce {mib:4d} MiB: {dt*1e3:8.1f} ms  "
                f"algbw {algbw:6.2f} GiB/s  busbw {busbw:6.2f} GiB/s",
                flush=True,
            )

    dist.barrier()
    if rank == 0:
        print("[SMOKE] PASS", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
