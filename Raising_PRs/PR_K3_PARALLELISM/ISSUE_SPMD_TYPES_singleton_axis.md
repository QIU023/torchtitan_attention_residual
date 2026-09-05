# Issue draft for meta-pytorch/spmd_types (0.2.5): typecheck crashes on a collective over a size-1 global axis

Not filed yet; the user decides. Found 2026-09-05 on torchtitan's Kimi K3 declarations under `--debug.spmd_typechecking`.

--- PASTE BEGIN ---

Title: typecheck(local=False): UnboundLocalError 'input_type' on redistribute over a size-1 axis

Under `typecheck(local=False)` any SPMD collective over a mesh axis of size 1 raises inside the checker instead of passing or reporting a type error:

```
  File ".../spmd_types/_checker/__init__.py", line 2865, in _typecheck_core
    if input_type is V and input_shard is None:
UnboundLocalError: cannot access local variable 'input_type' where it is not associated with a value
```

Repro (spmd_types 0.2.5, torch 2.11, gloo on CPU; `TP=1 torchrun --nproc_per_node=1 repro.py` fails, `TP=2 torchrun --nproc_per_node=2 repro.py` passes and types the result `I`):

```python
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
    print(dist.get_rank(), spmd.get_local_type(y))
```

Cause: in `_typecheck_core` the branch for SPMD collectives assigns `input_type` only while validating the axes resolved from the tensor's stored types (`for ax in resolved_axes: input_type = local_type[ax]`); a size-1 axis stores no type, so `resolved_axes` is `None` and the `elif axis.size() == 1: pass` branch skips the assignment. The global-axis block that follows (`if global_collective_axes:` ... `if input_type is V and input_shard is None:`) then reads the unassigned name. Under `typecheck(local=False)` every axis is global, so a model that keeps size-1 axes in its mesh (torchtitan's dense mesh does) cannot typecheck a conversion written for the general case.

Expected: the singleton axis is skipped in the global block as it is in the local one (the collective is an identity), or the `V` check only runs when a type was resolved.

--- PASTE END ---
