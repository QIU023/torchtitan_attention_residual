"""Local check (never committed): RemoteBackend round trip between two GPUs of one node.

torchrun --nproc_per_node=2 nvlink_intra_test.py; PROTO=tcp|nvlink_intra.
"""
import os

import torch
import torch.distributed as dist

from torchtitan.distributed.activation_storage import RemoteBackend


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    proto = os.environ.get("PROTO", "nvlink_intra")
    nbytes = 256 << 20
    before = torch.cuda.memory_allocated(dev)
    backend = RemoteBackend(
        dist.group.WORLD, dests={1: 0}, spans={1: nbytes}, staging_bytes=nbytes,
        device=dev, protocol=proto,
    )
    pool = (torch.cuda.memory_allocated(dev) - before) / 2**20
    print(f"[{proto}] rank {rank}: device memory taken by the backend {pool:.0f} MiB", flush=True)
    ok = True
    if rank == 1:
        stream = torch.cuda.Stream()
        x = torch.randn(nbytes // 8, device=dev)
        payload = backend.put(x, stream)
        torch.cuda.synchronize()
        out = torch.empty_like(x)
        backend.get(payload, out, stream)
        torch.cuda.synchronize()
        ok = torch.equal(out, x)
        print(f"[{proto}] rank 1: put {x.numel() * 4 >> 20} MiB, payload {payload}, round trip equal: {ok}", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
