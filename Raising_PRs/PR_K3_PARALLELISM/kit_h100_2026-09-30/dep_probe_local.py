"""Local probe recipe (never committed): the DEP ratio configs with the features spliced into stage 0 dumped.

With PROBE_FEATS_OUT set, every call of Kimi K3's scatter_vision_embeds on this rank saves its vision_embeds and the
spliced embeddings to <out>/rank<r>_call<n>.pt (the first PROBE_FEATS_MAX calls, default 8), then runs as before; and at
the first forward_backward each rank saves every parameter's and buffer's shape, sum and absolute sum (float64) of its
model parts to <out>/params_rank<r>.pt, before any update.
"""

import os

import torch
import torch.distributed as dist

import dep_ratio_local as base
from torchtitan.models.kimi_k3 import model as k3_model

_calls = [0]
_original = k3_model.scatter_vision_embeds


def _dumping(inputs_embeds, *, vision_embeds, vision_positions):
    out = _original(inputs_embeds, vision_embeds=vision_embeds, vision_positions=vision_positions)
    folder = os.environ.get("PROBE_FEATS_OUT")
    n = _calls[0]
    _calls[0] += 1
    if folder and n < int(os.environ.get("PROBE_FEATS_MAX", "8")):
        os.makedirs(folder, exist_ok=True)
        rank = dist.get_rank() if dist.is_initialized() else 0
        torch.save({"vision_embeds": vision_embeds.detach().cpu(), "embeds": out.detach().cpu(),
                    "positions": vision_positions}, os.path.join(folder, f"rank{rank}_call{n}.pt"))
    return out


k3_model.scatter_vision_embeds = _dumping


def _install_param_dump():
    from torch.distributed.tensor import DTensor

    from torchtitan import training_engine

    original = training_engine.TrainingEngine.forward_backward
    done = [False]

    def forward_backward(self, *args, **kwargs):
        folder = os.environ.get("PROBE_FEATS_OUT")
        if folder and not done[0]:
            done[0] = True
            os.makedirs(folder, exist_ok=True)
            rows = {}
            for i, part in enumerate(self.model_parts):
                for kind, items in (("param", part.named_parameters()), ("buffer", part.named_buffers())):
                    for name, t in items:
                        local = t.to_local() if isinstance(t, DTensor) else t
                        local = local.detach().double()
                        rows[f"part{i}.{kind}.{name}"] = (tuple(local.shape), local.sum().item(), local.abs().sum().item())
            rank = dist.get_rank() if dist.is_initialized() else 0
            torch.save(rows, os.path.join(folder, f"params_rank{rank}.pt"))
        return original(self, *args, **kwargs)

    training_engine.TrainingEngine.forward_backward = forward_backward


_install_param_dump()
w_dep_off = base.w_dep_off
w_dep_k25 = base.w_dep_k25
