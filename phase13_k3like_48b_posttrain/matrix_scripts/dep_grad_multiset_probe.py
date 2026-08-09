# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Are DEP-on and DEP-off computing the same gradients, or only the same loss?

From a shared seed checkpoint the two arms agree on step-1 loss to every printed
digit, so the forward is exact. grad_norm still differs by about 0.6%, which is
either a different summation partition in bf16 or a real difference in the
gradients themselves. Those need separating and the global norm cannot do it.

What can: the MULTISET of per-parameter gradient norms. DEP moves parameters to
different stages but does not change which parameters exist, so if the gradients
are the same values in a different arrangement, the multiset is identical and the
norm gap is pure reduction order. If the multiset differs, the gradients differ,
and the entries that fail to match name the shapes to look at.

No parameter names are needed for that, which is why it is done this way -- the
clip_grad_norm_ seam sees tensors, not names, and reaching for names would mean
touching the trainer.

Usage (any cell, both arms, same seed checkpoint):
    KIMI_VIT_DEP=1 torchrun --nproc_per_node=4 \\
      matrix_scripts/dep_grad_multiset_probe.py --module kimi_k3 --config <flavor> \\
      --checkpoint.initial_load_path <seed>/step-1 ...

Then compare the two arms' "[grad-multiset]" lines with --compare:
    python dep_grad_multiset_probe.py --compare off.log on.log
"""

from __future__ import annotations

import os
import sys


def _entries(params):
    """(numel, grad L2) per parameter that has a gradient, as printable text."""
    import torch

    out = []
    for p in params:
        g = p.grad
        if g is None:
            continue
        if hasattr(g, "to_local"):
            g = g.to_local()
        # Rounded to 6 significant digits: enough to separate real differences
        # from the last-bit noise of printing, and stable across runs because
        # the arms are deterministic.
        out.append(f"{g.numel()}:{float(torch.linalg.vector_norm(g.float())):.6g}:{g.dtype}")
    return out


def _compare(path_a: str, path_b: str) -> int:
    from collections import Counter

    def load(path):
        c = Counter()
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if "[grad-multiset]" not in line:
                    continue
                c.update(line.split("[grad-multiset]", 1)[1].split())
        return c

    a, b = load(path_a), load(path_b)
    print(f"{path_a}: {sum(a.values())} gradient entries, {len(a)} distinct")
    print(f"{path_b}: {sum(b.values())} gradient entries, {len(b)} distinct")
    only_a, only_b = a - b, b - a
    if not only_a and not only_b:
        print("MULTISETS IDENTICAL -- same gradients, so any grad_norm gap is reduction order")
        return 0
    print(f"DIFFER: {sum(only_a.values())} entries only in A, {sum(only_b.values())} only in B")
    for label, c in (("only in A", only_a), ("only in B", only_b)):
        for entry, n in sorted(c.items(), key=lambda kv: -kv[1])[:15]:
            numel, norm = entry.split(":")
            print(f"  {label}: numel={numel:>10} norm={norm} x{n}")
    return 1


def main() -> None:
    if "--compare" in sys.argv:
        i = sys.argv.index("--compare")
        raise SystemExit(_compare(sys.argv[i + 1], sys.argv[i + 2]))

    import torchtitan.distributed.utils as dist_utils

    orig = dist_utils.clip_grad_norm_
    rank = int(os.environ.get("RANK", "-1"))
    state = {"step": 0}

    def probe(parameters, *args, **kwargs):
        params = list(parameters)
        state["step"] += 1
        # BEFORE the clip, which rescales grads in place -- reading after it would
        # report whatever clipping normalized them to and hide the real difference.
        # Step 1 only: later steps have diverged weights, so their gradients are
        # expected to differ and would drown the signal.
        entries = _entries(params) if state["step"] == 1 else []
        out = orig(params, *args, **kwargs)
        if state["step"] == 1:
            print(
                f"[grad-multiset] " + " ".join(entries),
                flush=True,
            )
            print(
                f"[grad-multiset-rank {rank}] entries={len(entries)} "
                f"total_norm={float(out):.6f}",
                flush=True,
            )
        return out

    dist_utils.clip_grad_norm_ = probe

    import torchtitan.train as T

    if hasattr(T, "dist_utils"):
        T.dist_utils.clip_grad_norm_ = probe
    if hasattr(T, "clip_grad_norm_"):
        T.clip_grad_norm_ = probe
    T.main()


if __name__ == "__main__":
    main()
