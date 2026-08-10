"""Print a digest of every parameter right after the trainer builds the model.

Written to separate two explanations for a run-to-run difference: the arms compute
something different, or the arms START from different weights. A DEP arm builds the
vision tower into different stages, which changes the order modules are constructed
and therefore the order they draw from the RNG -- so a cold-initialized DEP-on run and
a cold DEP-off run can differ before a single forward has happened.

The digest is per-rank and order-independent within a rank (parameters are sorted by
name), so two runs on the same topology are directly comparable line by line. It is a
float sum rather than a hash because DTensor shards have to be reduced to something
comparable, and a sum of full_tensor() norms is enough to separate "same init" from
"different init".

Usage: torchrun ... init_digest_probe.py <the usual train args>
"""

from __future__ import annotations

import runpy

import torch
from torch.distributed.tensor import DTensor


def _digest(model_parts) -> str:
    total = 0.0
    count = 0
    for part in model_parts:
        for name, param in sorted(part.named_parameters(), key=lambda kv: kv[0]):
            tensor = param.full_tensor() if isinstance(param, DTensor) else param
            total += torch.linalg.vector_norm(
                tensor.detach(), 2, dtype=torch.float64
            ).item()
            count += 1
    return f"INIT_DIGEST params={count} norm_sum={total:.12e}"


def main() -> None:
    import torch.distributed as dist

    from torchtitan.trainer import Trainer

    original_init = Trainer.__init__

    def patched(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        rank = dist.get_rank() if dist.is_initialized() else 0
        print(f"[rank{rank}] {_digest(self.model_parts)}", flush=True)

    Trainer.__init__ = patched
    runpy.run_module("torchtitan.train", run_name="__main__")


if __name__ == "__main__":
    main()
