"""Does DTensor.scatter_ work on a token-sharded score tensor?

torchtitan/models/common/moe.py builds the routing map with

    torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(-1, topk_expert_ids_BLK, True)

and its comment asserts that "scatter_ writes along the (replicated) expert dim, so
DTensor runs it as a local op with no redistribution". This tests that assertion
directly on gloo/CPU -- no model, no GPU -- for the placement the EP x TP path produces:
scores sharded on the token dim.

Run: torchrun --nproc_per_node=2 scatter_repro.py
"""
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor

dist.init_process_group("gloo")
mesh = init_device_mesh("cpu", (dist.get_world_size(),), mesh_dim_names=("tp",))
rank = dist.get_rank()

B, L, E, K = 2, 8, 4, 2
torch.manual_seed(0)
scores = torch.randn(B, L, E)
ids = torch.randint(0, E, (B, L, K))


def report(name, fn):
    try:
        out = fn()
        placement = out.placements if isinstance(out, DTensor) else "plain"
        print(f"[rank{rank}] {name:34} OK   placements={placement}")
    except Exception as exc:  # noqa: BLE001 -- this is the measurement
        print(f"[rank{rank}] {name:34} RAISE {type(exc).__name__}: {str(exc)[:110]}")


for placements in ([Shard(1)], [Replicate()]):
    tag = str(placements[0])
    s = distribute_tensor(scores, mesh, placements)
    i = distribute_tensor(ids, mesh, placements)

    report(
        f"{tag}: in-place scatter_",
        lambda: torch.zeros_like(s, dtype=torch.bool).scatter_(-1, i, True),
    )
    report(
        f"{tag}: out-of-place scatter",
        lambda: torch.zeros_like(s, dtype=torch.bool).scatter(-1, i, True),
    )

if rank == 0:
    print(
        "\nWhat matters downstream: the routing map must keep Shard(1) so that "
        "routing_map.sum(dim=(0,1)) is Partial(sum) over the token axis."
    )
dist.destroy_process_group()
