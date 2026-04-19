# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Standalone smoke test for Block Attention Residuals.

Exercises ``block_attn_res`` and a simulated multi-layer / multi-block
forward pass using only ``torch`` (no torchtitan dependency chain). Runs
on CPU in a few seconds. Useful for sanity-checking the core logic when
the full torchtitan environment is not set up yet.

For the real unit tests (that import through torchtitan.models.*), see
``tests/unit_tests/test_attn_res.py`` -- those require a torchtitan-
compatible torch (>= 2.8 / nightly).
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F


# Inline the core primitive so we don't need torchtitan's import chain.
def block_attn_res(
    blocks: list[torch.Tensor],
    partial_block: torch.Tensor,
    proj: nn.Linear,
    norm: nn.Module,
) -> torch.Tensor:
    V = torch.stack(blocks + [partial_block], dim=0)
    K = norm(V)
    query = proj.weight.squeeze(0)
    logits = torch.einsum("d,nbtd->nbt", query, K)
    weights = F.softmax(logits, dim=0)
    return torch.einsum("nbt,nbtd->btd", weights, V)


def _zero_proj(dim: int) -> nn.Linear:
    proj = nn.Linear(dim, 1, bias=False)
    nn.init.zeros_(proj.weight)
    return proj


def _unit_norm(dim: int) -> nn.RMSNorm:
    return nn.RMSNorm(dim, eps=1e-5)


def test_core_primitive():
    torch.manual_seed(0)
    B, T, D = 2, 3, 8
    proj = _zero_proj(D)
    norm = _unit_norm(D)

    # 1. Single partial -> identity
    partial = torch.randn(B, T, D)
    out = block_attn_res([], partial, proj, norm)
    assert torch.allclose(out, partial, atol=1e-6), "single partial != identity"

    # 2. Zero query -> uniform average
    b0 = torch.randn(B, T, D)
    b1 = torch.randn(B, T, D)
    out = block_attn_res([b0, b1], partial, proj, norm)
    expected = (b0 + b1 + partial) / 3.0
    assert torch.allclose(out, expected, atol=1e-6), (
        f"zero-query not uniform: max diff={(out - expected).abs().max()}"
    )

    # 3. Gradient flow
    b0 = torch.randn(B, T, D, requires_grad=True)
    b1 = torch.randn(B, T, D, requires_grad=True)
    partial = torch.randn(B, T, D, requires_grad=True)
    out = block_attn_res([b0, b1], partial, proj, norm)
    out.sum().backward()
    assert b0.grad is not None and b1.grad is not None and partial.grad is not None
    # proj.weight at zero-init receives a non-zero grad when sources differ
    assert proj.weight.grad.abs().sum().item() > 0.0
    print("[PASS] core primitive: identity / uniform / gradient flow")


class MiniLayer(nn.Module):
    """Tiny stand-in for Llama3TransformerBlock with AttnRes.

    No attention / MLP -- just the block_attn_res bookkeeping. Attention
    and MLP are stubbed by identity + a learned linear, enough to verify
    the Decoder-level block threading.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.attn_res_proj = _zero_proj(dim)
        self.mlp_res_proj = _zero_proj(dim)
        self.attn_res_norm = _unit_norm(dim)
        self.mlp_res_norm = _unit_norm(dim)
        self.attn_norm = _unit_norm(dim)
        self.ffn_norm = _unit_norm(dim)
        self.attn_proj = nn.Linear(dim, dim, bias=False)
        self.ffn_proj = nn.Linear(dim, dim, bias=False)

    def forward_attn_res(self, blocks, partial_block, is_block_start):
        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)
        if is_block_start:
            blocks = blocks + [partial_block]
            partial_block = None
        attn_out = self.attn_proj(self.attn_norm(h))
        partial_block = attn_out if partial_block is None else partial_block + attn_out
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)
        mlp_out = self.ffn_proj(self.ffn_norm(h))
        partial_block = partial_block + mlp_out
        return blocks, partial_block


def test_multi_layer_flow():
    """Thread 6 layers / 3 blocks and verify block list grows correctly."""
    torch.manual_seed(1)
    B, T, D = 2, 4, 16
    n_layers = 6
    n_blocks = 3
    layers_per_block = n_layers // n_blocks  # 2

    layers = [MiniLayer(D) for _ in range(n_layers)]
    tok_emb = torch.randn(B, T, D, requires_grad=True)

    blocks = []
    partial = tok_emb
    for layer_id, layer in enumerate(layers):
        is_block_start = layer_id % layers_per_block == 0
        blocks, partial = layer.forward_attn_res(blocks, partial, is_block_start)

    # After 6 layers with block_start at 0, 2, 4: we committed 3 times.
    # Block 0 committed token_embedding (before any compute in layer 0).
    # Block 1 committed the accumulation of layers 0..1 output.
    # Block 2 committed the accumulation of layers 2..3 output.
    # After layer 5: partial_block = accumulation of layers 4..5.
    assert len(blocks) == n_blocks, f"expected {n_blocks} committed blocks, got {len(blocks)}"
    assert partial.shape == (B, T, D)
    print(f"[PASS] multi-layer flow: {n_blocks} committed blocks, final partial {tuple(partial.shape)}")

    # Backward reaches the token embedding through the whole chain.
    partial.sum().backward()
    assert tok_emb.grad is not None
    assert tok_emb.grad.abs().sum().item() > 0
    print("[PASS] backward reaches token embedding through the AttnRes chain")


def test_zero_init_equivalence_first_step():
    """At init, AttnRes should produce outputs close to standard residual.

    With all pseudo-queries zero, each block_attn_res returns a uniform
    average of its sources. This is not EXACTLY standard residual (which
    is a sum), but the hidden state at each layer is proportional to the
    standard residual hidden state. The key behavioral test: loss and
    gradients are well-defined and finite on the first step.
    """
    torch.manual_seed(2)
    B, T, D = 2, 4, 16
    n_layers = 4
    n_blocks = 2
    layers_per_block = n_layers // n_blocks

    layers = [MiniLayer(D) for _ in range(n_layers)]
    tok_emb = torch.randn(B, T, D, requires_grad=True)

    blocks = []
    partial = tok_emb
    for layer_id, layer in enumerate(layers):
        is_block_start = layer_id % layers_per_block == 0
        blocks, partial = layer.forward_attn_res(blocks, partial, is_block_start)

    # Final check: output is finite and non-NaN
    assert torch.isfinite(partial).all(), "output contains NaN or Inf"
    loss = partial.pow(2).mean()
    assert torch.isfinite(loss)
    loss.backward()
    # All layer params have finite gradients
    for i, layer in enumerate(layers):
        for name, p in layer.named_parameters():
            assert p.grad is not None, f"layer {i} param {name} has no grad"
            assert torch.isfinite(p.grad).all(), f"layer {i} param {name} grad has NaN/Inf"
    print("[PASS] zero-init: finite output / loss / gradients on all layer params")


def main():
    test_core_primitive()
    test_multi_layer_flow()
    test_zero_init_equivalence_first_step()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()
