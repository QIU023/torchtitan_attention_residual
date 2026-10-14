"""Logbook probe: dump the optimizer state, parameters and gradients around step 6's optimizer update, in a continuous
run and in a run resumed from the step-5 checkpoint, so the two can be compared tensor by tensor.

DUMP_DIR names the output folder; files are <tag>_rank<r>.pt with tags pre6 and post6.
"""
import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

import mm18_probe


def _local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def _dump(container, tag: str) -> None:
    from torchtitan.components.optim.utils import get_flat_optim_state_dict

    out = {}
    for i, optim in enumerate(container.optimizers):
        for key, value in get_flat_optim_state_dict(optim).items():
            out[f"opt{i}.{key}"] = _local(value).detach().clone().cpu() if isinstance(value, torch.Tensor) else value
        for group in optim.param_groups:
            for name, param in zip(group["param_names"], group["params"], strict=True):
                out[f"opt{i}.param.{name}"] = _local(param).detach().clone().cpu()
                if param.grad is not None:
                    out[f"opt{i}.grad.{name}"] = _local(param.grad).detach().clone().cpu()
    d = os.environ["DUMP_DIR"]
    os.makedirs(d, exist_ok=True)
    torch.save(out, f"{d}/{tag}_rank{dist.get_rank()}.pt")
    print(f"STATE_DUMP {tag} rank {dist.get_rank()} {len(out)} entries", flush=True)


def cell_resume_dump():
    from torchtitan.components.optim.optimizer import OptimizersContainer

    config = mm18_probe.cell_resume()
    if os.environ.get("STATIC_SHAPES") == "1":
        import torch._dynamo

        torch._dynamo.config.automatic_dynamic_shapes = False
        print("RESUME_PROBE: automatic_dynamic_shapes off", flush=True)
    if os.environ.get("DYN_FIRST") == "1":
        import torch._dynamo

        torch._dynamo.config.assume_static_by_default = False
        print("RESUME_PROBE: dynamic from the first compile", flush=True)
    resumed = int(os.environ.get("MM18_LOAD_STEP", "-1")) >= 0
    target = 1 if resumed else 6
    original = OptimizersContainer.step

    def step(self, *args, **kwargs):
        self._probe_calls = getattr(self, "_probe_calls", 0) + 1
        if self._probe_calls == target:
            _dump(self, "pre6")
        result = original(self, *args, **kwargs)
        if self._probe_calls == target:
            _dump(self, "post6")
        return result

    OptimizersContainer.step = step
    return config
