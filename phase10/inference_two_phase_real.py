"""Phase 10 Stage K — two-phase TP fabric pattern injected into real model inference.

Stage I demonstrated the RS+AG vs AllReduce fabric pattern with a
synthetic 12 MB tensor. Stage K does the same at **real-model scale**:
runs the actual kimi_linear AttnRes forward + injects RS+AG ops on
the per-attention-output tensor at each layer to fire the two-phase
fabric pattern that production-grade Block AttnRes inference would
generate.

This is a fabric demonstrator — it does NOT replace the model's
internal TP collective path (which still does standard AllReduce
inside o_proj's RowwiseParallel). The injected RS+AG ops add fabric
on top, producing the recognizable RS+AG pair signature in the trace
at real-model attention output shape.

Captures: real model fabric (FSDP AG + EP A2A + TP AR per layer per
forward) + injected (RS + AG per layer per forward) at real-model
attention output shape.

Run via phase10/run_two_phase_real.sh.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

_HERE = Path(__file__).resolve().parent
_WS = _HERE.parent
sys.path.insert(0, str(_WS))
sys.path.insert(0, str(_WS / "torchtitan"))

from phase10.inference_torchtitan import (  # noqa: E402
    _build_parallel_dims,
    _build_model_and_parallelize,
    _load_dcp,
    _init_dist,
)


def _inject_two_phase_fabric(
    attn_out_shape: tuple[int, ...],
    tp_group,
    n_layers: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Fire RS + local merge + AG ops matching attention-output shape.

    Mimics the post-attention TP collective path of two-phase Block
    AttnRes: ReduceScatter the attention output along seq, do a local
    merge (we use a no-op multiply here as proxy for online-softmax
    merge), then AllGather to re-assemble. Fired once per layer per
    forward call — equivalent to what would replace the standard
    o_proj's RowwiseParallel AllReduce in a production two-phase impl.
    """
    tp_world = dist.get_world_size(tp_group)
    # attn_out_shape: (B, T, D). RS along seq dim T (must be divisible
    # by tp_world). Pad if needed.
    B, T, D = attn_out_shape
    T_pad = ((T + tp_world - 1) // tp_world) * tp_world
    full_buf = torch.randn(
        B, T_pad, D, dtype=dtype, device=device,
    )
    rs_out = torch.empty(
        B, T_pad // tp_world, D, dtype=dtype, device=device,
    )
    ag_out = torch.empty_like(full_buf)
    for _ in range(n_layers):
        # Reshape so leading dim is divisible by tp_world for RS.
        flat = full_buf.reshape(B * T_pad, D).contiguous()
        rs_flat = rs_out.reshape(B * (T_pad // tp_world), D).contiguous()
        dist.reduce_scatter_tensor(rs_flat, flat, op=dist.ReduceOp.SUM, group=tp_group)
        # Local "merge" — proxy for Phase 2 online-softmax merge.
        merged = rs_flat * 1.0001
        ag_flat = ag_out.reshape(B * T_pad, D).contiguous()
        dist.all_gather_into_tensor(ag_flat, merged, group=tp_group)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--micro-bs", type=int, default=4)
    args = p.parse_args()

    _init_dist()
    rank = dist.get_rank()
    world = dist.get_world_size()
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    device = torch.device(f"cuda:{local}")

    if rank == 0:
        print(f"[two_phase_real] world={world} rank={rank}")

    pd = _build_parallel_dims(world)
    model = _build_model_and_parallelize(pd, device)
    _load_dcp(model, args.ckpt)

    # Get a TP-shape group. ParallelDims doesn't expose a sliceable
    # mesh_dim_names("tp",) on world_mesh in this version; build a
    # local sub-group of size pd.tp directly using the rank's tp slot.
    tp_size = pd.tp
    tp_rank_in_group = rank % tp_size
    tp_groups = []
    for grp_start in range(0, world, tp_size):
        ranks_in_grp = list(range(grp_start, grp_start + tp_size))
        g = dist.new_group(ranks=ranks_in_grp, backend="nccl")
        tp_groups.append(g)
    my_tp_group_idx = rank // tp_size
    tp_group = tp_groups[my_tp_group_idx]

    # kimi_linear 436M: 16 layers, hidden 1168.
    cfg_hidden = 1168
    n_layers = 16
    attn_out_shape = (args.micro_bs, args.seq_len, cfg_hidden)

    if rank == 0:
        print(
            f"[two_phase_real] tp_world={dist.get_world_size(tp_group)} "
            f"attn_shape={attn_out_shape} layers={n_layers}"
        )

    vocab = 163840
    rng = torch.Generator(device="cuda").manual_seed(42 + rank)
    t0 = time.time()
    with torch.no_grad():
        for step in range(1, args.n_steps + 1):
            ids = torch.randint(
                0, vocab, (args.micro_bs, args.seq_len),
                device=device, dtype=torch.long, generator=rng,
            )
            # Real model forward — fires standard inference fabric.
            out = model(ids)
            # Injected two-phase RS+AG fabric — ones per layer per step.
            _inject_two_phase_fabric(
                attn_out_shape, tp_group, n_layers,
                dtype=torch.bfloat16, device=device,
            )
            if rank == 0 and step % 10 == 0:
                elapsed = time.time() - t0
                print(f"[two_phase_real] step={step:3d}/{args.n_steps} t={elapsed:.1f}s")
            del out

    if rank == 0:
        print(f"[two_phase_real] DONE {args.n_steps} steps in {time.time()-t0:.1f}s")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
