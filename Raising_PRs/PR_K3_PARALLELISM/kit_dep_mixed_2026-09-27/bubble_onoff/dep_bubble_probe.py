"""Local probe (never committed): the vit_dep cell with the bubble on or off, dumping every
parameter's gradient right before each optimizer step, per rank, into $DEP_GRAD_DUMP."""

import os

import torch

from torchtitan.components.optimizer.optimizer import OptimizersContainer
from torchtitan.trainer import Trainer

from dep_bubble import dep_bubble_off, dep_bubble_on

_orig_step = OptimizersContainer.step
_calls = {"n": 0}


def _dumping_step(self, closure=None):
    _calls["n"] += 1
    out = os.environ["DEP_GRAD_DUMP"]
    rank = int(os.environ.get("RANK", "0"))
    grads = {}
    for part_idx, model in enumerate(self.model_parts):
        for name, p in model.named_parameters():
            if p.grad is None:
                continue
            g = p.grad
            if hasattr(g, "full_tensor"):
                g = g.full_tensor()
            grads[f"{part_idx}:{name}"] = g.detach().clone().cpu()
    torch.save(grads, os.path.join(out, f"rank{rank}_step{_calls['n']}.pt"))
    print(f"DEP_GRAD_DUMP rank {rank} step {_calls['n']}: {len(grads)} grads", flush=True)
    return _orig_step(self, closure)


OptimizersContainer.step = _dumping_step


def probe_on() -> Trainer.Config:
    return dep_bubble_on()


def probe_off() -> Trainer.Config:
    return dep_bubble_off()
