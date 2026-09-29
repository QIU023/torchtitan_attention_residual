"""Local probe (never committed): dep_new's configs, plus every parameter dumped right before
the first optimizer step (DEP_PARAM_DUMP), i.e. the weights the model starts from."""

import os

import torch

import dep_new
from dep_new import dep_bubble, dep_k25, dep_off  # noqa: F401
from torchtitan.components.optimizer.optimizer import OptimizersContainer

_inner = OptimizersContainer.step
_calls = {"n": 0}


def _param_dumping_step(self, closure=None):
    _calls["n"] += 1
    out = os.environ.get("DEP_PARAM_DUMP")
    if out and _calls["n"] == 1:
        rank = int(os.environ.get("RANK", "0"))
        params = {}
        for part_idx, model in enumerate(self.model_parts):
            for name, p in model.named_parameters():
                t = p.full_tensor() if hasattr(p, "full_tensor") else p
                params[f"{part_idx}:{name}"] = t.detach().clone().cpu()
        os.makedirs(out, exist_ok=True)
        torch.save(params, os.path.join(out, f"rank{rank}.pt"))
    return _inner(self, closure)


OptimizersContainer.step = _param_dumping_step
assert dep_new is not None
