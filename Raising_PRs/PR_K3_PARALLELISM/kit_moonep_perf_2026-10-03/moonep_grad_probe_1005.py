"""Step-1 gradient dumps for MoonEP item 12 (logbook kit only, never committed): the cells of moonep_probe_1003,
with OptimizersContainer.step patched to save every parameter's full gradient (fp32, rank 0) to PROBE_GRAD_OUT
before the first optimizer step."""

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from moonep_probe_1003 import *  # noqa: F401,F403
from torchtitan.components.optim.optimizer import OptimizersContainer

_step = OptimizersContainer.step
_dumped: list[bool] = []


def _dump_then_step(self, *args, **kwargs):
    out = os.environ.get("PROBE_GRAD_OUT")
    if out and not _dumped:
        grads = {}
        for part in self.model_parts:
            for name, param in part.named_parameters():
                grad = param.grad
                if grad is None:
                    continue
                if isinstance(grad, DTensor):
                    grad = grad.full_tensor()
                grads[name] = grad.detach().float().cpu()
        if dist.get_rank() == 0:
            torch.save(grads, out)
        _dumped.append(True)
    return _step(self, *args, **kwargs)


OptimizersContainer.step = _dump_then_step
