"""Review-tree probe (local kit, never committed): rt_cells with clipping off; every parameter's gradient at step 1
(before the update) and every parameter after the last update, as full tensors in DUMP_DIR/{grad1,param_last}_rank<r>.pt."""
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import rt_cells


def _dump(container, tag: str) -> None:
    out, dtypes = {}, set()
    for model in container.model_parts:
        for name, param in model.named_parameters():
            t = param.grad if tag == "grad1" else param
            if t is None:
                continue
            t = t.full_tensor() if isinstance(t, DTensor) else t
            out[name.replace("._checkpoint_wrapped_module", "")] = t.detach().cpu()
            dtypes.add(str(t.dtype))
    d = os.environ["DUMP_DIR"]
    os.makedirs(d, exist_ok=True)
    torch.save(out, f"{d}/{tag}_rank{dist.get_rank()}.pt")
    print(f"STATE_DUMP {tag} rank {dist.get_rank()} {len(out)} tensors dtypes {sorted(dtypes)}", flush=True)


def cell():
    from torchtitan.components.optim.optimizer import OptimizersContainer

    config = rt_cells.cell()
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


def seed():
    return rt_cells.seed()
