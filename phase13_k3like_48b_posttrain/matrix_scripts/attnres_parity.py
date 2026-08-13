"""Do the two Block-AttnRes implementations compute the same thing?

Both are now in the tree, so this is a direct comparison on identical inputs rather than a
reading of two sources. Two differences are visible in the code and only one is a pure
algebraic identity:

  query folding  theirs computes norm.weight * proj.weight once and contracts against the
                 UNWEIGHTED normalized keys; ours applies the norm (with weight) then
                 contracts. (x*rsqrt*w).q == (x*rsqrt).(w*q) -- identity.
  cast order     theirs floats BEFORE computing variance/rsqrt; ours normalizes in the
                 stream dtype and floats after. NOT an identity in bf16, and our own
                 comment says the release does v.float() first.

So: expect a match in fp32 and a difference in bf16, with theirs closer to the release.
"""
import torch
from torchtitan.models.kimi_k3.attn_res import block_attn_res
from torchtitan.models.kimi_k3_up.model import _apply_attention_residual

torch.manual_seed(0)
B, T, D, N = 2, 16, 64, 3

class _Proj(torch.nn.Module):
    def __init__(self, w): super().__init__(); self.weight = torch.nn.Parameter(w)

class _Norm(torch.nn.Module):
    def __init__(self, w, eps=1e-6):
        super().__init__(); self.weight = torch.nn.Parameter(w); self.eps = eps
    def forward(self, x):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + self.eps) * self.weight

for dtype in (torch.float32, torch.bfloat16):
    blocks = [torch.randn(B, T, D, dtype=dtype) for _ in range(N)]
    partial = torch.randn(B, T, D, dtype=dtype)
    proj = _Proj(torch.randn(1, D, dtype=dtype) * 0.02)
    norm = _Norm(torch.ones(D, dtype=dtype))

    ours = block_attn_res(blocks, partial, proj, norm)

    # theirs takes [T', N, D] with T' = B*L flattened and the prefix sum separate
    block_residual_TND = torch.stack([b.reshape(-1, D) for b in blocks], dim=1)
    theirs = _apply_attention_residual(
        partial.reshape(-1, D), block_residual_TND, proj, norm
    ).view(B, T, D)

    diff = (ours.float() - theirs.float()).abs().max().item()
    scale = ours.float().abs().max().item()
    print(f"{str(dtype):16} max abs diff {diff:.3e}   relative {diff / max(scale, 1e-12):.3e}")
