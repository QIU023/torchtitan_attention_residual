"""Do the two Block-AttnRes implementations compute the same thing?

Both are in the tree, so this is a direct comparison on identical inputs rather than a
reading of two sources. Two differences are visible in the code and only one is a pure
algebraic identity:

  query folding  theirs computes norm.weight * proj.weight once and contracts against the
                 UNWEIGHTED normalized keys; ours applies the norm (with weight) then
                 contracts. (x*rsqrt*w).q == (x*rsqrt).(w*q) -- identity, so the norm
                 weight must be RANDOM here or this arm is not exercised at all.
  cast order     the release floats BEFORE computing variance/rsqrt. Ours used to
                 normalize in the stream dtype and float after, which is not an identity
                 in bf16 -- 3.6e-3 relative, fixed 2026-08-14.

Expected: both dtypes at ~1e-7 relative (reduction order only). The third column is the
pre-fix form, kept so the regression cannot return silently -- it should stay at ~1e-3
in bf16, which is what this probe was written to catch.
"""
# NOTE (2026-08-19): this probe imports the VENDORED upstream K3 tree, which has been
# deleted -- torchtitan/models/kimi_k3_up/ existed to be diffed against, and the pr4025
# git remote does that without a copy that drifts. Two of the four probes that used it
# were already broken by the 2026-08-15 rollback that stripped our work out of the
# vendored tree, and nothing noticed.
#
# To run this again, restore the tree from history:
#     git -C torchtitan checkout 0cadf15e0 -- torchtitan/models/kimi_k3_up
# and re-add "kimi_k3_up" to the registry list in torchtitan/models/__init__.py.
# Prefer pinning to their CURRENT head instead, since the vendored copy was already
# three reuse commits behind by the time it was deleted:
#     git -C torchtitan show pr4025/agent/add-kimi-k3-reference-model:<path>

import torch
import torch.nn.functional as F

from torchtitan.models.kimi_k3.attn_res import block_attn_res
from torchtitan.models.kimi_k3_up.model import _apply_attention_residual

torch.manual_seed(0)
B, T, D, N = 2, 16, 64, 3


class _Proj(torch.nn.Module):
    def __init__(self, w):
        super().__init__()
        self.weight = torch.nn.Parameter(w)


class _Norm(torch.nn.Module):
    def __init__(self, w, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(w)
        self.eps = eps

    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.weight


def _pre_fix(blocks, partial, proj, norm):
    """block_attn_res as it stood before 2026-08-14: norm in the stream dtype."""
    V = torch.stack(blocks + [partial], dim=0)
    K = norm(V).float()
    query = proj.weight.squeeze(0).float()
    weights = F.softmax(torch.einsum("d,nbtd->nbt", query, K), dim=0)
    return torch.einsum("nbt,nbtd->btd", weights, V.float()).to(V.dtype)


def _rel(a, b):
    diff = (a.float() - b.float()).abs().max().item()
    return diff / max(a.float().abs().max().item(), 1e-12)


print(f"{'dtype':16} {'current':>12} {'pre-fix':>12}")
for dtype in (torch.float32, torch.bfloat16):
    blocks = [torch.randn(B, T, D, dtype=dtype) for _ in range(N)]
    partial = torch.randn(B, T, D, dtype=dtype)
    proj = _Proj(torch.randn(1, D, dtype=dtype) * 0.02)
    # NOT ones: a unit weight makes the query-folding arm vacuous.
    norm = _Norm(1.0 + 0.1 * torch.randn(D, dtype=dtype))

    # theirs takes [T', N, D] with T' = B*L flattened and the prefix sum separate
    block_residual_TND = torch.stack([b.reshape(-1, D) for b in blocks], dim=1)
    theirs = _apply_attention_residual(
        partial.reshape(-1, D), block_residual_TND, proj, norm
    ).view(B, T, D)

    ours = block_attn_res(blocks, partial, proj, norm)
    old = _pre_fix(blocks, partial, proj, norm)
    print(f"{str(dtype):16} {_rel(theirs, ours):>12.3e} {_rel(theirs, old):>12.3e}")
