"""Measure cancellation inside block_attn_res, and dump full gradients.

The AttnRes pseudo-query gradient is grad_query = sum_{n,b,t} grad_logits * K.
If the softmax is near-uniform (proj is zero-init, so it starts exactly uniform),
grad_logits = w * (dL - sum(w*dL)) is a difference of nearly equal quantities and
the sum cancels. Under cancellation an absolute perturbation that is negligible
against the TERMS becomes large against the RESULT, which is what a percent-level
tp1-vs-tp2 ratio on this parameter would mean if nothing is actually broken.

Reports, for the first block_attn_res call:
  sum|term|   the scale of what is being summed
  |sum term|  the surviving result
  ratio       the cancellation factor

and dumps the full gradient of the AttnRes projections so tp1 and tp2 can be
differenced in absolute terms, not just by norm ratio.
"""

from __future__ import annotations

import os
import runpy
import sys

import torch


def _install() -> None:
    import torch.distributed as dist

    import torchtitan.experiments.kimi_k3.attn_res_model as M
    from torchtitan.experiments.kimi_k3.attn_res import block_attn_res as _orig

    seen: list[int] = []

    def patched(blocks, partial_block, proj, norm):
        out = _orig(blocks, partial_block, proj, norm)
        if len(seen) < 6 and out.requires_grad:
            seen.append(1)
            V = torch.stack(blocks + [partial_block], dim=0)
            K = norm(V).to(V.dtype)

            def _hook(g_out):
                # dL/dh at the AttnRes output; reconstruct the terms whose sum
                # forms grad_query, without re-running autograd.
                w = torch.softmax(
                    torch.einsum(
                        "d,nbtd->nbt",
                        proj.weight.detach().to(K.dtype).squeeze(0),
                        K,
                    ),
                    dim=0,
                )
                dl = torch.einsum("nbtd,btd->nbt", V.detach(), g_out.detach())
                gl = w * (dl - (w * dl).sum(dim=0, keepdim=True))
                terms = torch.einsum("nbt,nbtd->nbtd", gl, K.detach())
                flat = terms.reshape(-1, terms.shape[-1]).float()
                sum_abs = flat.abs().sum(dim=0).norm().item()
                abs_sum = flat.sum(dim=0).norm().item()
                rank = dist.get_rank() if dist.is_initialized() else 0
                print(
                    f"[CANCEL] rank={rank} call#{len(seen)} sum|term|={sum_abs:.6e} "
                    f"|sum term|={abs_sum:.6e} "
                    f"cancellation_factor={sum_abs / max(abs_sum, 1e-30):.3e}",
                    flush=True,
                )
                return g_out

            out.register_hook(_hook)
        return out

    M.block_attn_res = patched

    import torchtitan.distributed.utils as du

    orig_clip = du.clip_grad_norm_

    def clip(parameters, *a, **k):
        ps = list(parameters)
        path = os.environ.get("GRADTENSOR_DUMP")
        if path:
            rank = dist.get_rank() if dist.is_initialized() else 0
            if rank == 0:
                keep = {}
                for m in _PARTS[0] if _PARTS else []:
                    for n, p in m.named_parameters():
                        if "res_proj" in n and p.grad is not None:
                            g = p.grad
                            g = g.full_tensor() if hasattr(g, "full_tensor") else g
                            keep[n] = g.detach().float().cpu()
                torch.save(keep, path)
                print(f"[CANCEL] dumped {len(keep)} grads -> {path}", flush=True)
            du.clip_grad_norm_ = orig_clip
        return orig_clip(ps, *a, **k)

    du.clip_grad_norm_ = clip


_PARTS: list = []


def _grab() -> None:
    import torchtitan.train as T

    orig_init = T.Trainer.__init__

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        _PARTS.append(self.model_parts)

    T.Trainer.__init__ = init


_install()
_grab()
sys.argv[0] = "torchtitan.train"
runpy.run_module("torchtitan.train", run_name="__main__", alter_sys=True)
