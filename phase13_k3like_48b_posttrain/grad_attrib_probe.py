# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-parameter gradient attribution across one varied parallelism dimension.

The instrument this project keeps needing. A loss curve cannot see a gradient
that is wrong on one parameter family, or on a handful of boundary positions,
or only once an adapter stops being zero -- every defect found here passed a
loss comparison first.

Three properties matter and all three have been got wrong before:

* **Warm, not cold.** LoRA initializes B to zero, so at step 0
  ``grad_A = grad_out @ B^T`` is exactly zero and a cold check measures an inert
  adapter. Run some steps first, checkpoint, and start every leg from that.
* **Shared start.** FSDP2 meta-init gives each parallelism layout its own RNG
  draw, so two layouts started cold are not comparable at all.
* **One varied dimension.** Anything else and the attribution is ambiguous.

Usage: run once per leg, then diff the dumps.

    torchrun --nproc_per_node=1 grad_attrib_probe.py --config <flavor> \
        --load <warm-checkpoint> --out /tmp/g_tp1.pt
    torchrun --nproc_per_node=2 grad_attrib_probe.py --config <flavor> \
        --load <warm-checkpoint> --out /tmp/g_tp2.pt \
        --parallelism.tensor_parallel_degree 2
    python grad_attrib_probe.py --diff /tmp/g_tp1.pt /tmp/g_tp2.pt

The diff reports ``ratio = |g_ref| / |g_leg|`` per parameter, so a value above 1
means the varied leg's gradient is too SMALL -- a missing reduction -- and below
1 means too large. Both directions have shown up: block_attn_res over-reduced by
exactly 1/tp, moe_sharding dropped a reduction entirely.
"""

from __future__ import annotations

import argparse
import sys

import torch


def _norms(model) -> dict[str, object]:
    """Per-parameter gradients, gathered to full tensors where sharded.

    Keeps the tensor, not just its norm. A norm ratio is blind to direction and
    unstable on small values -- it reported |ratio-1| up to 2.1 on parameters
    whose gradients are three orders of magnitude below the model median, which
    is floating-point cancellation rather than a defect. Cosine similarity
    between the two gradients answers the question the ratio cannot: is this the
    same gradient computed differently, or a different gradient?
    """
    out: dict[str, object] = {}
    for name, p in model.named_parameters():
        g = p.grad
        if g is None:
            out[name] = None
            continue
        if hasattr(g, "full_tensor"):
            g = g.full_tensor()
        out[name] = g.detach().float().flatten().cpu()
    return out


def diff(ref_path: str, leg_path: str, top: int = 25) -> int:
    ref = torch.load(ref_path)
    leg = torch.load(leg_path)
    shared = sorted(set(ref) & set(leg))
    if not shared:
        print("no parameters in common", file=sys.stderr)
        return 1

    rows = []
    zero_ref = []
    frozen = []
    for name in shared:
        ta, tb = ref[name], leg[name]
        if ta is None and tb is None:
            frozen.append(name)
            continue
        if ta is None or tb is None:
            rows.append((float("inf"), 0.0, name, 0.0, 0.0))
            continue
        a = float(ta.norm()); b = float(tb.norm())
        cos = float(torch.nn.functional.cosine_similarity(ta, tb, dim=0)) if a and b else 0.0
        if a == 0.0:
            zero_ref.append(name)
            continue
        rows.append((a / b if b else float("inf"), cos, name, a, b))
        continue
        if ta != ta and tb != tb:
            # No gradient on either side. Legitimately frozen -- LoRA trains
            # only the adapters -- so this is not a finding.
            frozen.append(name)
            continue
        if a != a or b != b:
            # A gradient on one side only. That IS a finding: the parameter is
            # trainable in one layout and inert in the other.
            rows.append((float("inf"), name, a, b))
            continue
        if a == 0.0:
            zero_ref.append(name)
            continue
        rows.append((a / b if b else float("inf"), name, a, b))

    if frozen:
        print(f"(skipped {len(frozen)} parameter(s) with no gradient in either "
              "leg -- frozen base weights)\n")

    if zero_ref:
        # Exactly the failure mode a cold run produces. Say it loudly rather
        # than reporting a clean-looking result over the parameters that moved.
        print(
            f"WARNING: {len(zero_ref)} parameter(s) have a ZERO reference "
            f"gradient and were skipped, e.g. {zero_ref[:3]}. If these are "
            "adapter or gate parameters, the reference run was cold and this "
            "comparison measures nothing.\n"
        )

    # Rank by 1-cos: direction disagreement, which small magnitudes do not fake.
    rows.sort(key=lambda r: -(1.0 - r[1]))
    print(f"{'1-cos':>10}  {'ratio':>9}  {'|g|ref':>12}  parameter")
    for ratio, cos, name, a, b in rows[:top]:
        print(f"{1.0 - cos:>10.2e}  {ratio:>9.4f}  {a:>12.6g}  {name}")

    cosr = [r[1] for r in rows if r[0] != float("inf")]
    ratios = [abs(r[0] - 1.0) for r in rows if r[0] != float("inf")]
    med = sorted(ratios)
    print(
        f"\nworst 1-cos   = {1.0 - min(cosr) if cosr else float('nan'):.3e}"
        f"\nmax |ratio-1| = {max(ratios) if ratios else 0.0:.5f}   median = "
        f"{med[len(med)//2] if med else float('nan'):.5f}   over {len(rows)} params"
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--diff", nargs=2, metavar=("REF", "LEG"))
    ap.add_argument("--out")
    known, rest = ap.parse_known_args()

    if known.diff:
        return diff(*known.diff)

    # Training mode: run one step through the real trainer, then dump.
    import torchtitan.train as titan_train

    dumped = {}
    orig_step = titan_train.Trainer.train_step

    def train_step(self, *a, **k):
        result = orig_step(self, *a, **k)
        for part in self.model_parts:
            dumped.update(_norms(part))
        raise SystemExit(0)  # one step is all we need

    titan_train.Trainer.train_step = train_step
    sys.argv = [sys.argv[0]] + rest
    try:
        titan_train.main()
    except SystemExit:
        pass
    if known.out:
        # Per-rank filename. Under PP each rank holds a different stage, so a
        # single path is both incomplete AND racy -- the first version of this
        # produced a 27 MB corrupted archive from concurrent writers.
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        path = known.out if rank == 0 else f"{known.out}.rank{rank}"
        torch.save(dumped, path)
        print(f"[grad-attrib] rank {rank}: {len(dumped)} parameters -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
