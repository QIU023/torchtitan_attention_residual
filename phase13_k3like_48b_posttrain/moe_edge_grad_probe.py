"""Measure the gradient on the ONE tensor the MoE tp defect lives on.

The parameter-level view cannot separate "the tp ranks disagree" from "the ranks
agree but a reduction is dropped" -- both show up as a wrong parameter gradient.
This hooks the tensor between latent.down and the experts (the MoE's latent
input) and reports its gradient norm ON EVERY RANK, so the two cases look
different: disagreement shows as ranks differing from each other, a dropped
reduction shows as every rank agreeing with each other but not with tp1.

Usage: torchrun ... moe_edge_grad_probe.py <the usual train args>
"""

from __future__ import annotations

import runpy
import sys

import torch


def _install() -> None:
    import torch.distributed as dist

    from torchtitan.experiments.kimi_k3.model import KimiMoE

    original = KimiMoE.forward
    seen: list[int] = []

    def patched(self, x):
        if self.latent_size is not None and not seen:
            latent_in = self.latent.to_latent(x)

            def _hook(g):
                rank = dist.get_rank() if dist.is_initialized() else 0
                import os
                if os.environ.get("MOEEDGE_ALLREDUCE") and dist.is_initialized():
                    # If the true gradient is the SUM of the per-rank shares,
                    # all-reducing here must reproduce the tp1 value exactly.
                    g = g.clone()
                    dist.all_reduce(g)
                print(
                    f"[MOEEDGE] rank={rank} grad_latent_input_norm="
                    f"{g.detach().float().pow(2).sum().sqrt().item():.8f} "
                    f"shape={tuple(g.shape)} type={type(g).__name__}",
                    flush=True,
                )
                return g

            if latent_in.requires_grad:
                latent_in.register_hook(_hook)
                seen.append(1)
            # K3 passes a SECOND input the sharding config never declares:
            # the router reads x while the experts consume W_down x. Hook it
            # too -- if its gradient is Partial and declared Replicate, the
            # ranks disagree and all-reducing recovers the tp1 value.
            router_in = x
            if router_in.requires_grad:

                def _rhook(g):
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    import os
                    if os.environ.get("MOEEDGE_ALLREDUCE") and dist.is_initialized():
                        g = g.clone()
                        dist.all_reduce(g)
                    print(
                        f"[MOEEDGE] rank={rank} grad_ROUTER_input_norm="
                        f"{g.detach().float().pow(2).sum().sqrt().item():.8f}",
                        flush=True,
                    )
                    return g

                router_in.register_hook(_rhook)
            out = self._moe(latent_in, router_input_BLD=router_in)
            if isinstance(out, torch.Tensor) and hasattr(out, "placements"):
                print(
                    f"[MOEEDGE] moe_out placements={out.placements} "
                    f"mesh={out.device_mesh.mesh_dim_names}",
                    flush=True,
                )
            from torch.distributed.tensor import DTensor, Replicate

            def _mk(tag):
                def _h(g):
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    import os
                    if os.environ.get("MOEEDGE_ALLREDUCE") and dist.is_initialized():
                        g = g.clone()
                        dist.all_reduce(g)
                    gg = g.to_local() if hasattr(g, "to_local") else g
                    print(
                        f"[MOEEDGE] rank={rank} grad_{tag}="
                        f"{gg.detach().float().pow(2).sum().sqrt().item():.8f}",
                        flush=True,
                    )
                    return g
                return _h

            if out.requires_grad:
                out.register_hook(_mk("MOE_OUT_pre_to_local"))
            if isinstance(out, DTensor):
                if any(not p.is_replicate() for p in out.placements):
                    out = out.redistribute(
                        placements=[Replicate()] * len(out.placements)
                    )
                out = out.to_local()
            if out.requires_grad:
                out.register_hook(_mk("MOE_OUT_post_to_local"))
            out = self.latent.from_latent(out)
            if self.shared_experts is not None:
                out = out + self.shared_experts(x)
            return out
        return original(self, x)

    KimiMoE.forward = patched


_install()
sys.argv[0] = "torchtitan.train"
runpy.run_module("torchtitan.train", run_name="__main__", alter_sys=True)
