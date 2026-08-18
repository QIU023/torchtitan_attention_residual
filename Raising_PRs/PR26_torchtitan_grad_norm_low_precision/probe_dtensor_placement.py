"""Does passing ``dtype=float32`` to the norm preserve DTensor's ``_NormPartial``?

PR26's helper docstring claims "the dtype argument preserves ``_NormPartial``". That line
was read out of the source, never run, and if it is wrong the PR body is wrong: a norm
whose placement drops to plain or to ``Partial(sum)`` reduces ACROSS RANKS incorrectly, so
the reported grad_norm would be wrong in exactly the distributed setting the PR exists to
fix.

What ``_NormPartial`` is for: ``vector_norm`` of a ``Shard(0)`` DTensor produces one local
norm per rank, and combining them is NOT a sum -- for p=2 it is ``sqrt(sum of squares)``.
DTensor encodes that pending reduction as ``_NormPartial`` so the next op knows to combine
with the norm rule rather than adding. If ``dtype=`` silently returns a plain tensor or a
``Partial(sum)``, the cross-rank total is computed the wrong way.

    torchrun --nproc_per_node=2 probe_dtensor_placement.py

gloo + CPU on purpose: no model, no CUDA, so it runs while the GPU is busy and reproduces
on any box. The property under test is placement algebra, which is device-independent.
"""

import os

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, DTensor, Replicate, Shard

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29561")
dist.init_process_group("gloo")
rank = dist.get_rank()
world = dist.get_world_size()
mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("dp",))


def show(tag, t):
    if rank != 0:
        return
    if isinstance(t, DTensor):
        print(
            f"  {tag}: DTensor dtype={t.dtype} placements={t.placements} "
            f"local={tuple(t.to_local().shape)}",
            flush=True,
        )
    else:
        print(f"  {tag}: {type(t).__name__} dtype={t.dtype} shape={tuple(t.shape)}", flush=True)


# A full gradient every rank agrees on, so a single-rank fp32 reference is exact.
torch.manual_seed(0)
full = torch.randn(8 * world, 16, dtype=torch.bfloat16)
reference_fp32 = torch.linalg.vector_norm(full.float(), 2.0).item()

shard = distribute_tensor(full, mesh, [Shard(0)])
rep = distribute_tensor(torch.randn(4, 16, dtype=torch.bfloat16), mesh, [Replicate()])

if rank == 0:
    print(f"world={world}  reference fp32 norm of the sharded grad = {reference_fp32:.6f}")
    print("\n[1] vector_norm of a Shard(0) DTensor, no dtype vs dtype=float32:")

n_plain = torch.linalg.vector_norm(shard, 2.0)
show("no dtype   ", n_plain)
n_fp32 = torch.linalg.vector_norm(shard, 2.0, dtype=torch.float32)
show("dtype=fp32 ", n_fp32)

# The claim, made checkable: both must stay DTensor, and the fp32 one must keep whatever
# placement the no-dtype one has -- that placement is what makes the next op reduce across
# ranks correctly. Print the class name of the placement so _NormPartial is visible by name.
if rank == 0:
    def pk(t):
        return type(t.placements[0]).__name__ if isinstance(t, DTensor) else "PLAIN"

    print(f"\n  placement no-dtype  : {pk(n_plain)}")
    print(f"  placement dtype=fp32: {pk(n_fp32)}")
    same = isinstance(n_fp32, DTensor) and pk(n_fp32) == pk(n_plain)
    print(f"  -> dtype preserves the placement: {same}")

# [2] End to end through the helper the PR ships, on a DTensor list. DTensors force the
# per-tensor path (upstream's foreach check excludes them), so this exercises exactly the
# branch the claim is about, then reduces to a scalar the way clip_grad_norm_ consumes it.
from torchtitan.distributed.utils import _get_total_norm_fp32  # noqa: E402

if rank == 0:
    print("\n[2] _get_total_norm_fp32 on [Shard(0), Replicate()] DTensors:")
total = _get_total_norm_fp32([shard, rep], 2.0, error_if_nonfinite=False, foreach=None)
show("total_norm ", total)

# clip_grad_norm_ calls .full_tensor() on a DTensor result; do the same and compare the
# SHARDED-tensor contribution against its fp32 reference. Not bitwise: fp32-over-bf16
# narrows the grouping difference, it does not erase it (trap 2 in the handoff).
if isinstance(total, DTensor):
    total_full = total.full_tensor().item()
else:
    total_full = total.item()

# Reference including the replicated tensor, computed once on the full data.
rep_full = rep.full_tensor() if isinstance(rep, DTensor) else rep
ref_both = torch.linalg.vector_norm(
    torch.stack(
        [
            torch.linalg.vector_norm(full.float(), 2.0),
            torch.linalg.vector_norm(rep_full.float(), 2.0),
        ]
    ),
    2.0,
).item()

if rank == 0:
    rel = abs(total_full - ref_both) / max(ref_both, 1e-12)
    print(f"  total (full_tensor) = {total_full:.6f}   fp32 reference = {ref_both:.6f}")
    print(f"  relative error = {rel:.3e}")
    ok = same and isinstance(total, DTensor) and rel < 1e-2
    print(f"\nPROBE {'PASS' if ok else 'FAIL'}: "
          f"placement preserved AND cross-rank total within 1% of fp32 reference")

dist.destroy_process_group()
