"""CPU probe, logbook kit only (gloo, 8 ranks, pp2 x cp4): install_vision_cp builds the sub-CP groups on every rank when
the model config turns dynamic CP on, including the pipeline stage that holds no tower, and builds none when it is off."""

import datetime

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.models.kimi_k3 import build_model_config
from torchtitan.models.kimi_k3.vision_cp import install_vision_cp


class _Context:
    cp_enabled = True

    def __init__(self, cp_mesh):
        self._cp_mesh = cp_mesh

    def get_mesh(self, name):
        assert name == "cp", name
        return self._cp_mesh


class _Tower:
    subgroups = None

    def set_cp_subgroups(self, subgroups):
        self.subgroups = subgroups


dist.init_process_group("gloo", timeout=datetime.timedelta(seconds=120))
rank = dist.get_rank()
mesh = init_device_mesh("cpu", (2, 4), mesh_dim_names=("pp", "cp"))
stage = mesh["pp"].get_local_rank()
context = _Context(mesh["cp"])
config = build_model_config("debugmodel").vision_encoder
assert config.dynamic_cp_min_patches is None, config.dynamic_cp_min_patches
for name, min_patches in (("off (default)", None), ("on (256)", 256)):
    config.dynamic_cp_min_patches = min_patches
    tower = _Tower() if stage == 0 else None
    install_vision_cp(tower, config, context)
    dist.barrier()
    if tower is not None:
        got = None if tower.subgroups is None else {k: dist.get_process_group_ranks(g) for k, g in tower.subgroups.items()}
        print(f"rank {rank} stage 0 {name}: tower sub-groups {got}", flush=True)
    else:
        print(f"rank {rank} stage 1 {name}: no tower, returned", flush=True)
dist.barrier()
if rank == 0:
    print("PP_PROBE_OK", flush=True)
dist.destroy_process_group()
