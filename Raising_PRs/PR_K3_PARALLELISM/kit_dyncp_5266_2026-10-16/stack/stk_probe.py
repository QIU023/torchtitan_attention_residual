"""Stack probe (local kit, never committed): combo_cells with clipping off; the stage-0 vision tower's gradients at step 1
(before the update) and its parameters after the last update, as full tensors in DUMP_DIR/{grad1,param_last}_rank<r>.pt,
and the FSDP mesh of the tower's parameters."""
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import combo_cells


def _dump(container, tag: str) -> None:
    out, mesh = {}, None
    for model in container.model_parts:
        for name, param in model.named_parameters():
            if "vision_encoder" not in name:
                continue
            if mesh is None and isinstance(param, DTensor):
                mesh = (param.device_mesh.mesh_dim_names, param.placements)
            t = param.grad if tag == "grad1" else param
            if t is None:
                continue
            t = t.full_tensor() if isinstance(t, DTensor) else t
            out[name.replace("._checkpoint_wrapped_module", "")] = t.detach().float().cpu()
    if not out:
        return
    d = os.environ["DUMP_DIR"]
    os.makedirs(d, exist_ok=True)
    torch.save(out, f"{d}/{tag}_rank{dist.get_rank()}.pt")
    print(f"STATE_DUMP {tag} rank {dist.get_rank()} {len(out)} tensors tower mesh {mesh}", flush=True)


def cell():
    from torchtitan.components.optim.optimizer import OptimizersContainer

    config = combo_cells.cell()
    config.optim.max_norm = 1.0e9
    original_step = OptimizersContainer.step

    def step(self, *args, **kwargs):
        self._probe_calls = getattr(self, "_probe_calls", 0) + 1
        dump = os.environ.get("DUMP_DIR")
        if self._probe_calls == 1 and dump:
            _dump(self, "grad1")
        result = original_step(self, *args, **kwargs)
        if self._probe_calls == int(os.environ.get("K3C_STEPS", "10")) and dump:
            _dump(self, "param_last")
        return result

    OptimizersContainer.step = step
    return config
