"""#4380 diagnosis (logbook kit only): cpmm_probe's cell with clip_grad_norm_ wrapped to print, on rank 0, the device
meshes its gradients fall into (titan computes one norm per mesh and combines them) and how many parameters each holds."""
import os
from collections import Counter

import torch.distributed as dist
from torch.distributed.tensor import DTensor

import cpmm_probe


def cell():
    import torchtitan.distributed.utils as dist_utils

    original = dist_utils.clip_grad_norm_
    calls = [0]

    def clip(parameters, *args, **kwargs):
        parameters = list(parameters)
        if calls[0] == 0 and dist.get_rank() == 0:
            groups = Counter()
            tower = Counter()
            for p in parameters:
                if p.grad is None:
                    continue
                mesh = p.grad.device_mesh if isinstance(p.grad, DTensor) else None
                key = "local" if mesh is None else f"{mesh.mesh_dim_names}{tuple(mesh.shape)}"
                groups[key] += 1
                tower[key] += int(getattr(p, "_meshlog_tower", False))
            print(f"MESHLOG {len(groups)} mesh group(s): " + "; ".join(f"{k}: {n} params, {tower[k]} of them tower" for k, n in groups.items()), flush=True)
        calls[0] += 1
        return original(parameters, *args, **kwargs)

    dist_utils.clip_grad_norm_ = clip
    config = cpmm_probe.cell()
    return config


def _mark_tower():
    from torchtitan.training_engine import TrainingEngine

    original = TrainingEngine.optim_step

    def optim_step(self):
        if self.num_completed_steps == 0:
            for part in self.model_parts:
                for name, param in part.named_parameters():
                    param._meshlog_tower = "vision_encoder" in name
        return original(self)

    TrainingEngine.optim_step = optim_step


_mark_tower()
