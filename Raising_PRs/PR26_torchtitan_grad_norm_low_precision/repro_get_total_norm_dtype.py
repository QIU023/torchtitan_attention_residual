# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`get_total_norm`'s total depends on how the tensors are grouped. Repro at 1/2/4/8 ranks.

    python repro_get_total_norm_dtype.py                      # CPU/gloo, world 1,2,4,8
    python repro_get_total_norm_dtype.py --device cuda        # CUDA/nccl
    python repro_get_total_norm_dtype.py --module <patched clip_grad.py>

The same 512 bf16 gradients at every world size; the only thing that changes is how they
are split across ranks. Two splits, both of which occur in real training:

  dtensor  each gradient is a DTensor Shard(0) over the mesh -- what FSDP produces.
           get_total_norm returns a DTensor carrying _NormPartial, and full_tensor()
           performs the cross-rank combine.
  pp       rank r owns gradients r::world as plain tensors -- what a pipeline stage
           holds. Each rank norms its own, then the partials are all-reduced.

The cross-rank reduction is the real one, in the dtype the partials are in. There is no
emulation and nothing is combined in float64 -- an earlier version of this evidence
combined the partials in float64 to isolate the per-group rounding, which made the
grouping dependence easy to see and also made the experiment synthetic.

The reference is float64 over the whole gradient set, computed identically on every rank,
so it does not depend on the world size. Neither should the reported norm.

Self-spawns rather than using torchrun: torchrun's rendezvous wants a libuv TCPStore,
which Windows torch builds lack, and a FileStore has no such dependency.
"""

import argparse
import importlib.util
import os
import tempfile

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard


N_GRADS = 512
NUMEL = 128
SEED = 0


def load_patched(path):
    if path is None:
        return None
    spec = importlib.util.spec_from_file_location("_patched_clip_grad", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._get_total_norm


def make_grads(device):
    """The same 512 gradients on every rank and at every world size."""
    torch.manual_seed(SEED)
    return [
        torch.randn(NUMEL, dtype=torch.bfloat16, device=device) for _ in range(N_GRADS)
    ]


def truth(grads):
    return torch.linalg.vector_norm(
        torch.cat([g.to(torch.float64).flatten() for g in grads]), 2
    ).item()


def dtensor_total(get_total_norm, grads, mesh, dtype=None):
    """FSDP-shaped split: every gradient sharded over the mesh."""
    kwargs = {} if dtype is None else {"dtype": dtype}
    dt = [distribute_tensor(g, mesh, [Shard(0)]) for g in grads]
    total = get_total_norm(dt, 2.0, **kwargs)
    return total.full_tensor().item() if hasattr(total, "full_tensor") else total.item()


def pp_total(get_total_norm, grads, rank, world, dtype=None):
    """Pipeline-shaped split: rank r owns gradients r::world, partials all-reduced."""
    kwargs = {} if dtype is None else {"dtype": dtype}
    local = get_total_norm(grads[rank::world], 2.0, **kwargs)
    squared = local.detach().clone().float() ** 2
    dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    return squared.sqrt().item()


def worker(rank, world, device_type, module_path, init_file, out):
    dist.init_process_group(
        backend="nccl" if device_type == "cuda" else "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world,
    )
    if device_type == "cuda":
        torch.cuda.set_device(rank % torch.cuda.device_count())
        device = f"cuda:{rank % torch.cuda.device_count()}"
    else:
        device = "cpu"
    mesh = init_device_mesh(device_type, (world,))

    stock = torch.nn.utils.get_total_norm
    patched = load_patched(module_path)
    grads = make_grads(device)

    values = {
        "dtensor today": dtensor_total(stock, grads, mesh),
        "pp today": pp_total(stock, grads, rank, world),
    }
    if patched is not None:
        values["dtensor fp32"] = dtensor_total(
            patched, grads, mesh, dtype=torch.float32
        )
        values["pp fp32"] = pp_total(patched, grads, rank, world, dtype=torch.float32)
    if rank == 0:
        values["truth"] = truth(grads)
        out.update(values)
    dist.destroy_process_group()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    ap.add_argument("--module", default=None, help="path to a patched clip_grad.py")
    ap.add_argument("--world", default="1,2,4,8")
    args = ap.parse_args()

    worlds = [int(w) for w in args.world.split(",")]
    if args.device == "cuda":
        avail = torch.cuda.device_count()
        worlds = [w for w in worlds if w <= avail] or [avail]

    print(f"torch {torch.__version__}  device={args.device}  "
          f"patched={'yes' if args.module else 'no'}")

    manager = mp.Manager()
    results = {}
    for world in worlds:
        out = manager.dict()
        with tempfile.TemporaryDirectory() as d:
            init_file = os.path.join(d, "store").replace("\\", "/")
            mp.spawn(
                worker,
                args=(world, args.device, args.module, init_file, out),
                nprocs=world,
                join=True,
            )
        results[world] = dict(out)

    ref = next(iter(results.values()))["truth"]
    keys = [k for k in ("dtensor today", "dtensor fp32", "pp today", "pp fp32")
            if k in next(iter(results.values()))]
    width = max(len(k) for k in keys)
    header = "  ".join(f"world={w}".rjust(11) for w in worlds)
    print(f"\n{'':<{width}}  {header}      spread")
    for key in keys:
        vals = [results[w][key] for w in worlds]
        spread = (max(vals) - min(vals)) / max(vals)
        cells = "  ".join(f"{v:11.6f}" for v in vals)
        print(f"{key:<{width}}  {cells}   {spread:.2e}")
    print(f"\nfloat64 truth (world-independent by construction): {ref:.6f}")
    return 0


if __name__ == "__main__":
    main()
