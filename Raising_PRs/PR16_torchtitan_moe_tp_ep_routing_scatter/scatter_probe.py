"""Minimal probe: does in-place scatter_ on a Shard(1) DTensor error on THIS torch?

Isolates the PR16 question from any model/trainer. Windows-safe: FileStore
rendezvous (no TCPStore/libuv). Run:  python scatter_probe.py
"""
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank: int, world: int, store_path: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    store = dist.FileStore(store_path, world)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("tp",))

    torch.manual_seed(0)
    scores_full = torch.randn(2, 8, 4)          # B, L, E
    topk_full = torch.randint(0, 4, (2, 8, 2))  # B, L, K
    scores = distribute_tensor(scores_full, mesh, [Shard(1)])
    topk = distribute_tensor(topk_full, mesh, [Shard(1)])

    try:
        m = torch.zeros_like(scores, dtype=torch.bool).scatter_(
            -1, topk, True
        )
        ref = torch.zeros_like(scores_full, dtype=torch.bool).scatter_(
            -1, topk_full, True
        )
        ok_place = m.placements == scores.placements
        ok_value = torch.equal(m.full_tensor(), ref)
        counts = m.to_local().sum(dim=(0, 1)).tolist()
        print(
            f"[rank{rank}] torch={torch.__version__} SCATTER OK  "
            f"placements_kept={ok_place} value_correct={ok_value} "
            f"local_counts={counts}",
            flush=True,
        )
    except Exception as e:
        print(
            f"[rank{rank}] torch={torch.__version__} SCATTER ERROR  "
            f"{type(e).__name__}: {str(e)[:300]}",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    path = os.path.join(tempfile.mkdtemp(), "store")
    mp.spawn(worker, args=(2, path), nprocs=2, join=True)
