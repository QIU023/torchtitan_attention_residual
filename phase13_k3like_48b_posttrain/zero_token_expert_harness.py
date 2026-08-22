"""Force a routed expert to receive zero tokens and check training survives.

Issue 4269's Gate 5 asks for zero-token experts. They do not occur naturally on
the small flavours: the smallest sequence KDA accepts is T > 64, and eight
experts at top-2 over that many tokens still give every expert tens of them
(measured: min 17 at seq 128, min 519 at seq 4096). So the case has to be
constructed.

This drives the real training entrypoint and pushes one expert's routing score
to -inf through a forward pre-hook on the router, which is what the load-balance
bias does anyway -- expert_bias_E is added to the scores before top-k. The hook
lives here rather than in the tree so the matrix cannot be affected by it.

Usage (4 GPUs):
  torchrun --nproc_per_node=4 zero_token_expert_harness.py -- <train args...>
"""

from __future__ import annotations

import os
import sys

import torch


def install_zero_token_bias(model_parts, banned_expert: int = 0) -> int:
    """Add -inf to one expert's bias on every MoE. Returns the number patched."""
    patched = 0
    for part in model_parts:
        for module in part.modules():
            bias = getattr(module, "expert_bias_E", None)
            if bias is None:
                continue
            target = bias.to_local() if hasattr(bias, "to_local") else bias
            with torch.no_grad():
                target[banned_expert] = float("-inf")
            patched += 1
    return patched


def main() -> None:
    if "--" in sys.argv:
        sys.argv = [sys.argv[0]] + sys.argv[sys.argv.index("--") + 1 :]
    from torchtitan.train import main as train_main
    import torchtitan.trainer as trainer_mod

    banned = int(os.environ.get("K3_BANNED_EXPERT", "0"))
    seen = {"counts": None}

    original_train_step = trainer_mod.Trainer.train_step

    def patched(self, *args, **kwargs):
        if not getattr(self, "_zt_installed", False):
            self._zt_installed = True
            n = install_zero_token_bias(self.model_parts, banned)
            if torch.distributed.get_rank() == 0:
                print(f"[ZT] banned expert {banned} on {n} MoE modules", flush=True)
            for part in self.model_parts:
                for module in part.modules():
                    if type(module).__name__ != "KimiMoE":
                        continue
                    router = getattr(getattr(module, "_moe", None), "router", None)
                    if router is None:
                        continue

                    def hook(mod, inp, out, _s=seen):
                        ids = out[1]
                        ids = ids.to_local() if hasattr(ids, "to_local") else ids
                        cnt = torch.bincount(ids.reshape(-1), minlength=8)
                        _s["counts"] = cnt
                        return out

                    router.register_forward_hook(hook)
                    break
        result = original_train_step(self, *args, **kwargs)
        if seen["counts"] is not None and torch.distributed.get_rank() == 0:
            c = seen["counts"]
            print(
                f"[ZT] per-expert tokens: {c.tolist()} zero={int((c == 0).sum())}",
                flush=True,
            )
            seen["counts"] = None
        return result

    trainer_mod.Trainer.train_step = patched
    train_main()


if __name__ == "__main__":
    main()
