"""Goal probe (local kit, never committed): dump every rank's vision tower parameters and gradients around step 1's
optimizer update, with dynamic CP on or off, gradient clipping disabled. Builds on combo_cells (K3C_* settings).
DUMP_DIR names the output folder; files are pre1_rank<r>.pt and post1_rank<r>.pt."""
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import combo_cells


def _full(t):
    return t.full_tensor() if isinstance(t, DTensor) else t


def _dump(container, tag: str) -> None:
    out = {}
    for model in container.model_parts:
        for name, param in model.named_parameters():
            if "vision_encoder" not in name:
                continue
            name = name.replace("._checkpoint_wrapped_module", "")
            out[f"param.{name}"] = _full(param).detach().float().cpu()
            if param.grad is not None:
                out[f"grad.{name}"] = _full(param.grad).detach().float().cpu()
    d = os.environ["DUMP_DIR"]
    os.makedirs(d, exist_ok=True)
    torch.save(out, f"{d}/{tag}_rank{dist.get_rank()}.pt")
    print(f"STATE_DUMP {tag} rank {dist.get_rank()} {len(out)} entries", flush=True)


def cell():
    from torchtitan.components.optim.optimizer import OptimizersContainer

    config = combo_cells.cell()
    config.optim.max_norm = 1.0e9
    original_step = OptimizersContainer.step

    def step(self, *args, **kwargs):
        self._probe_calls = getattr(self, "_probe_calls", 0) + 1
        if self._probe_calls == 1:
            _dump(self, "pre1")
        result = original_step(self, *args, **kwargs)
        if self._probe_calls == 1:
            _dump(self, "post1")
        return result

    OptimizersContainer.step = step
    return config
