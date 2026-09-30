"""Local probe recipe (never committed): the DEP ratio configs with the features spliced into stage 0 dumped.

With PROBE_FEATS_OUT set, every call of Kimi K3's scatter_vision_embeds on this rank saves its vision_embeds and the
spliced embeddings to <out>/rank<r>_call<n>.pt (the first PROBE_FEATS_MAX calls, default 8), then runs as before.
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
w_dep_off = base.w_dep_off
w_dep_k25 = base.w_dep_k25
