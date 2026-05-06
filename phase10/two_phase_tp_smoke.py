"""Phase 10 Stage I — two-phase TP fabric pattern demo.

Captures the unique fabric signature of Block AttnRes's two-phase
computation (per the project blog): the post-attention TP collective
path fuses Phase 2's online softmax merge into a
``ReduceScatter -> local merge -> AllGather`` pattern instead of the
standard single ``AllReduce``.

Volume-wise the two patterns are equivalent (RS half-msg + AG half-msg
== full-msg AR). Fabric-pattern-shape-wise they differ:

* baseline: 1 AllReduce of full-message-size, nranks=TP
* two-phase: 1 ReduceScatter + 1 AllGather of half-message-size each,
  nranks=TP

This script captures both patterns in one process (mode controlled by
``--mode`` arg) so the fabric trace pipeline produces two
ixia_config.json files that can be loaded into IXIA side-by-side for
visual comparison.

Run::

    bash phase10/run_two_phase_smoke.sh
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("allreduce", "rs_ag"), required=True)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--per-attn-msg-mb", type=float, default=12.0,
                   help="Full-message size in MB matching v11 attention TP "
                        "AllReduce (~12 MB at 436M). RS+AG uses half-msg per call.")
    p.add_argument("--n-attn-per-step", type=int, default=16,
                   help="One per layer. v11 has 16 layers.")
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl")

    # Use the world group as TP group for this smoke (TP=world_size).
    tp_group = dist.group.WORLD

    full_bytes = int(args.per_attn_msg_mb * 1024 * 1024)
    full_floats = full_bytes // 2  # bf16
    # For RS+AG: shard along the leading axis. We size the buffer so
    # that the reduce-scatter output is exactly half-msg per call,
    # matching the blog's "Phase 2 split" semantics.
    if args.mode == "allreduce":
        buf = torch.randn(full_floats, dtype=torch.bfloat16, device="cuda")
    else:
        # RS expects buf to be evenly divisible by tp_world_size.
        # We emit a buf of size full_floats * tp (so RS produces full_floats
        # chunks per rank, AG re-assembles the same total size).
        buf = torch.randn(full_floats * world, dtype=torch.bfloat16, device="cuda")
        rs_out = torch.empty(full_floats, dtype=torch.bfloat16, device="cuda")
        ag_out = torch.empty(full_floats * world, dtype=torch.bfloat16, device="cuda")

    if rank == 0:
        print(
            f"[smoke] mode={args.mode} world={world} per_attn_mb={args.per_attn_msg_mb} "
            f"layers={args.n_attn_per_step} steps={args.n_steps}"
        )

    t0 = time.time()
    for step in range(1, args.n_steps + 1):
        for layer in range(args.n_attn_per_step):
            if args.mode == "allreduce":
                dist.all_reduce(buf, op=dist.ReduceOp.SUM, group=tp_group)
            else:
                # ReduceScatter: each rank reduces buf, scatters to rs_out.
                dist.reduce_scatter_tensor(
                    rs_out, buf, op=dist.ReduceOp.SUM, group=tp_group,
                )
                # "Local merge" — represents the online softmax merge step.
                # In the real Block AttnRes path this is elementwise; here
                # we just multiply by a constant so the result depends on
                # rs_out (so the AG below isn't optimized away).
                merged = rs_out * 1.0001
                # AllGather: re-assemble across TP ranks.
                dist.all_gather_into_tensor(ag_out, merged, group=tp_group)

        if rank == 0 and step % 10 == 0:
            print(f"[smoke] step={step}/{args.n_steps} t={time.time()-t0:.2f}s")

    if rank == 0:
        print(f"[smoke] DONE mode={args.mode} {args.n_steps} steps in {time.time()-t0:.2f}s")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
