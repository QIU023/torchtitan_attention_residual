# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Do replicated parameters hold the SAME gradient on every TP rank?

A parameter marked Replicate is supposed to carry an identical value on every
rank of the mesh. Its gradient only stays identical if every rank's backward
saw identical inputs; where the surrounding computation is sharded it does not,
and the per-rank gradients are then partial contributions that must be summed.
Skip that reduction and each rank keeps its own fraction -- no error, no shape
mismatch, just a wrong update.

``full_tensor()`` cannot detect this: for a Replicate placement it returns one
rank's copy without checking the others agree. This probe compares the ranks
directly.

    torchrun --nproc_per_node=2 replicated_grad_rank_probe.py \
        --module kimi_k3 --config <flavor> \
        --parallelism.tensor_parallel_degree 2 ...
"""

from __future__ import annotations

import sys

import torch
import torch.distributed as dist


def _report(model_parts, tp_group=None) -> None:
    world = dist.get_world_size(tp_group)
    rank = dist.get_rank(tp_group)
    findings: list[tuple[float, str, float]] = []
    checked = 0

    for part in model_parts:
        for name, p in part.named_parameters():
            g = p.grad
            if g is None:
                continue
            placements = getattr(g, "placements", None)
            local = g.to_local() if hasattr(g, "to_local") else g
            # Check the TP axis only. Under FSDP every parameter is Shard on
            # the dp axis, so requiring Replicate on all axes excludes
            # everything and the probe silently checks nothing.
            group = tp_group
            if placements is not None:
                mesh = g.device_mesh
                names = mesh.mesh_dim_names or ()
                if "tp" not in names:
                    continue
                axis = names.index("tp")
                if not placements[axis].is_replicate():
                    continue
                group = mesh.get_group("tp")
            if group is None:
                continue
            checked += 1
            local = local.detach().float()
            gathered = [
                torch.empty_like(local) for _ in range(dist.get_world_size(group))
            ]
            dist.all_gather(gathered, local.contiguous(), group=group)
            if rank != 0:
                continue
            base = gathered[0]
            spread = max(
                (gathered[r] - base).abs().max().item()
                for r in range(1, len(gathered))
            )
            scale = base.abs().max().item()
            if scale > 0 and spread / scale > 1e-3:
                findings.append((spread / scale, name, scale))

    if rank != 0:
        return
    findings.sort(reverse=True)
    print(f"\n[rank-spread] checked {checked} replicated-gradient parameters")
    if not findings:
        print("[rank-spread] all agree across ranks")
        return
    print(f"[rank-spread] {len(findings)} DISAGREE across ranks:")
    print(f"{'rel spread':>12}  {'|g|max':>12}  parameter")
    for rel, name, scale in findings[:25]:
        print(f"{rel:>12.4f}  {scale:>12.6g}  {name}")
    print(
        "\nA replicated parameter whose gradient differs by rank is missing its "
        "all-reduce: each rank holds a partial contribution and will apply only "
        "its own share."
    )


def main() -> int:
    import torchtitan.train as titan_train

    orig = titan_train.Trainer.train_step

    def train_step(self, *a, **k):
        out = orig(self, *a, **k)
        _report(self.model_parts)
        raise SystemExit(0)

    titan_train.Trainer.train_step = train_step
    try:
        titan_train.main()
    except SystemExit:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
