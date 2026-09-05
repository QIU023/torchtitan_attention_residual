# spmd_types 0.2.5: redistribute over a size-1 global axis under typecheck(local=False)
# raises UnboundLocalError('input_type') in _typecheck_core; TP=2 is the control.
import os, torch, torch.distributed as dist
import spmd_types as spmd
from spmd_types.checker import typecheck
from torch.distributed.device_mesh import init_device_mesh
dist.init_process_group("gloo")
tp = int(os.environ.get("TP", "1"))
mesh = init_device_mesh("cpu", (dist.get_world_size() // tp, tp), mesh_dim_names=("dp", "tp"))
with spmd.set_current_mesh(mesh), typecheck(local=False):
    x = torch.zeros(4, 0, 8)
    y = spmd.redistribute(x, mesh.get_group("tp"), src=spmd.R, dst=spmd.I)
    print(f"rank {dist.get_rank()} tp={tp} ok {tuple(y.shape)} type={spmd.get_local_type(y)}", flush=True)
dist.destroy_process_group()
