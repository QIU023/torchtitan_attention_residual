# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The partitioned vision tower against the replicated one, on two ranks.

    STATE 2026-09-22: rewritten from the lost 09-10 scratch version and NOT yet
    working -- both ranks reach the partitioned call and then sit at 0% GPU with
    the memory held, which is a collective the two ranks do not both reach. Debug
    before quoting any number from it.

    torchrun --nproc_per_node=2 vitcp_probe.py [--dtype float32|bfloat16]

Both ranks build the same tower from the same seed. The replicated path runs the
whole image on each rank; the partitioned path goes through the model's own
``encode_images``, which cuts the image over the CP group. The forward is
compared directly, and the gradients are compared as the sum over ranks of the
partitioned parameter gradients against the replicated gradient, since every
rank scores the whole gathered output.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.kimi_k3.config_registry import model_registry
from torchtitan.models.kimi_k3.parallelize import _build_cp_subgroups


def _grid(t: int, h: int, w: int, device) -> torch.Tensor:
    return torch.tensor([[t, h, w]], dtype=torch.long, device=device)


def _case(name: str, t: int, h: int, w: int):
    return name, t, h, w


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", default="float32")
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    device = torch.device("cuda", rank % torch.cuda.device_count())
    mesh = init_device_mesh("cuda", (1, dist.get_world_size()), mesh_dim_names=("dp", "cp"))

    spec = model_registry("debugmodel")
    model_config = spec.model
    torch.manual_seed(0)
    model = model_config.build().to(device=device, dtype=dtype)
    model.set_vision_cp_subgroups(_build_cp_subgroups(mesh.get_group("cp")))
    model_config.dynamic_cp_min_patches = 1
    model.dynamic_cp_min_patches = 1
    encoder = model.vision_encoder
    assert encoder is not None
    kh, kw = encoder.merge_kernel_size
    channels = encoder.patch_embed.weight.shape[-1]

    rows = []
    for name, t, h, w in [
        _case("one image", 1, 4 * kh, 4 * kw),
        _case("padded band", 1, 3 * kh, 4 * kw),
        _case("video t=2", 2, 4 * kh, 4 * kw),
    ]:
        torch.manual_seed(1234)
        pixels = torch.randn(t * h * w, channels, device=device, dtype=dtype)
        grid = _grid(t, h, w, device)

        model.zero_grad(set_to_none=True)
        with set_current_spmd_mesh(mesh):
            ref = encoder(pixels, grid_thw=grid)
        torch.manual_seed(99)
        weight = torch.randn_like(ref)
        (ref * weight).sum().backward()
        ref_grads = {n: p.grad.detach().clone() for n, p in encoder.named_parameters() if p.grad is not None}

        model.zero_grad(set_to_none=True)
        with set_current_spmd_mesh(mesh):
            out = model.encode_images(pixels, grid)
        ((out * weight).sum() / dist.get_world_size()).backward()
        summed = {}
        for n, p in encoder.named_parameters():
            g = p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            dist.all_reduce(g)
            summed[n] = g

        fwd = (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-12)
        rels = []
        for n, g in summed.items():
            r = ref_grads.get(n)
            if r is None:
                continue
            rels.append(((g - r).abs().max() / r.abs().max().clamp_min(1e-12)).item())
        rels.sort()
        if rank == 0:
            median = rels[len(rels) // 2] if rels else float("nan")
            rows.append((name, fwd, median, rels[-1] if rels else float("nan")))

    if rank == 0:
        print(f"dtype={args.dtype} ranks={dist.get_world_size()}")
        print("| case | forward max diff / scale | grad rel, median | grad rel, worst |")
        print("| --- | ---: | ---: | ---: |")
        for name, fwd, med, worst in rows:
            print(f"| {name} | {fwd:.1e} | {med:.1e} | {worst:.1e} |")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
