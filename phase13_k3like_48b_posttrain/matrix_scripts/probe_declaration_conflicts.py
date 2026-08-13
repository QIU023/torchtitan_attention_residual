"""Where do the imperative plan and the declarations disagree about a placement?

Read-only: after apply_tp has run, compare every declared state_sharding against what the
parameter actually holds. _distribute_states raises on a mismatch, so this lists all of
them at once instead of one per crash.
"""
import torch
from torch.distributed.tensor import DTensor
from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.protocols.module import Module as TTModule
from torchtitan.protocols.sharding import resolve_placements

torch.distributed.init_process_group("nccl")
world, rank = torch.distributed.get_world_size(), torch.distributed.get_rank()
torch.cuda.set_device(rank)
pd = ParallelDims(dp_replicate=1, dp_shard=1, cp=1, tp=world, pp=1, ep=1, world_size=world)
pd.build_mesh()

from torchtitan.models.kimi_k3.config_registry import kimi_k3_debugmodel_report_arch
with torch.device("cuda"):
    model = kimi_k3_debugmodel_report_arch().model_spec.model.build()

from torchtitan.models.kimi_k3.parallelize import apply_tp_kimi_k3
apply_tp_kimi_k3(model, pd.get_mesh("tp"), None, None)

if rank == 0:
    agree = mismatch = undeclared = plain = 0
    for name, m in model.named_modules():
        sc = getattr(m, "_sharding_config", None)
        if sc is None or not isinstance(m, TTModule):
            continue
        for pname, p in m.named_parameters(recurse=False):
            layout = (sc.state_shardings or {}).get(pname)
            if layout is None:
                undeclared += 1
                print(f"[undeclared] {name}.{pname}")
                continue
            if not isinstance(p, DTensor):
                plain += 1
                continue
            mesh = pd.resolve_mesh(layout.axes())
            if mesh is None:
                continue
            expected = resolve_placements(layout, mesh)
            if tuple(p.placements) == tuple(expected):
                agree += 1
            else:
                mismatch += 1
                print(f"[MISMATCH] {name}.{pname}: has {p.placements}, declared {expected}")
    print(f"[probe] agree={agree} mismatch={mismatch} undeclared={undeclared} still-plain={plain}")
torch.distributed.destroy_process_group()
