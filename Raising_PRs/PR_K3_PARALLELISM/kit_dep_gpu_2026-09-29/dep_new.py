"""Local recipes (never committed): the new DEP cell (k3_pp_mm bb3e38d4a) with DEP off, on in K2.5 mode,
and on with the bubble, all with the even data-parallel ranks text only as the committed cell has it.
DEPN_MEM_OUT records every rank's per-step peak; DEP_GRAD_DUMP (with DEP_GRAD_STEPS) dumps every
parameter's gradient right before the optimizer step."""

import atexit
import json
import os

import torch

from torchtitan.components.optimizer.optimizer import OptimizersContainer
from torchtitan.trainer import Trainer

_RECORDS: list[dict] = []
_orig_step = OptimizersContainer.step
_calls = {"n": 0}


def _dumping_step(self, closure=None):
    _calls["n"] += 1
    out = os.environ.get("DEP_GRAD_DUMP")
    steps = {int(s) for s in os.environ.get("DEP_GRAD_STEPS", "1").split(",")}
    if out and _calls["n"] in steps:
        rank = int(os.environ.get("RANK", "0"))
        grads = {}
        for part_idx, model in enumerate(self.model_parts):
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                g = p.grad.full_tensor() if hasattr(p.grad, "full_tensor") else p.grad
                grads[f"{part_idx}:{name}"] = g.detach().clone().cpu()
        os.makedirs(out, exist_ok=True)
        torch.save(grads, os.path.join(out, f"rank{rank}_step{_calls['n']}.pt"))
    return _orig_step(self, closure)


OptimizersContainer.step = _dumping_step


def _record():
    _RECORDS.append({"max_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                     "max_reserved_gib": torch.cuda.max_memory_reserved() / 2**30})


def _install_memory():
    out = os.environ.get("DEPN_MEM_OUT")
    if not out:
        return
    from torchtitan.observability import metrics

    reset = metrics.DeviceMemoryMonitor.reset_peak_stats

    def reset_after_recording(self):
        _record()
        reset(self)

    metrics.DeviceMemoryMonitor.reset_peak_stats = reset_after_recording

    def dump():
        if not torch.cuda.is_available():
            return
        _record()
        rank = int(os.environ.get("RANK", "0"))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, f"rank{rank}.json"), "w") as f:
            json.dump({"rank": rank, "records": _RECORDS}, f)

    atexit.register(dump)


_install_memory()


def _base() -> Trainer.Config:
    from torchtitan_recipes.tests.b200 import (
        kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4_vision_dep,
    )

    return kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4_vision_dep()


def dep_off() -> Trainer.Config:
    config = _base()
    config.model.vision_dep.enabled = False
    config.model.vision_dep.bubble = False
    return config


def dep_k25() -> Trainer.Config:
    config = _base()
    config.model.vision_dep.bubble = False
    assert config.model.vision_dep.enabled
    return config


def dep_bubble() -> Trainer.Config:
    config = _base()
    assert config.model.vision_dep.enabled and config.model.vision_dep.bubble
    return config
