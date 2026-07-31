"""Count how many tokens change expert assignment when tp changes.

Routing is discrete: topk over the router scores is not continuous, so a
bf16-level difference in a score can flip a token to a different expert and
change that token's gradient discontinuously. That is not a placement bug and no
placement fix applies to it. This dumps the top-k expert ids so tp1 and tpN can
be compared token by token.

Zero flips alongside a gradient gap means a placement issue remains.
Flips means the gap has a discrete source and the parameter-level ratio is not
measuring a defect.
"""

from __future__ import annotations

import os
import runpy
import sys

import torch


def _install() -> None:
    import torch.distributed as dist

    from torchtitan.models.common.moe import TokenChoiceTopKRouter

    original = TokenChoiceTopKRouter.forward
    dumped: list[int] = []

    def patched(self, *a, **kw):
        out = original(self, *a, **kw)
        path = os.environ.get("ROUTEDUMP")
        if path and not dumped:
            dumped.append(1)
            sc = out[2] if len(out) > 2 else None
            if sc is not None:
                sc_l = sc.to_local() if hasattr(sc, "to_local") else sc
                sc_f = sc_l.detach().float()
                top = sc_f.topk(k=4, dim=-1).values
                gap = (top[..., 1] - top[..., 2]).abs()
                print(f"[ROUTE] score spread: max={sc_f.max():.6f} "
                      f"min={sc_f.min():.6f} "
                      f"median gap between rank-2 and rank-3 expert="
                      f"{gap.median():.3e} "
                      f"frac(gap<1e-3)={((gap<1e-3).float().mean()):.4f}",
                      flush=True)
            # forward returns (topk_scores_BLK, topk_expert_ids_BLK, scores_BLE)
            ids = out[1]
            ids = ids.to_local() if hasattr(ids, "to_local") else ids
            rank = dist.get_rank() if dist.is_initialized() else 0
            torch.save(ids.detach().cpu(), f"{path}.r{rank}")
            print(f"[ROUTE] rank={rank} ids{tuple(ids.shape)} -> {path}.r{rank}",
                  flush=True)
        return out

    TokenChoiceTopKRouter.forward = patched


_install()
sys.argv[0] = "torchtitan.train"
runpy.run_module("torchtitan.train", run_name="__main__", alter_sys=True)
