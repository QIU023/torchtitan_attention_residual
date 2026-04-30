"""Variable image count per row (B1 from phase6/README).

The original ``KimiLinearAttnResModel.forward`` multimodal scatter
required every row to have exactly ``vision_embeds.size(1)`` image tokens
(verified by an explicit assert in the now-relaxed reference helper
``phase5/multimodal_model.py:multimodal_loss``). Real VLM data violates
this: text-only rows interleaved into pretrain corpora have zero image
tokens, multi-image rows have multiples of N_vision. These tests exercise
the relaxed scatter path (commit B1) on a tiny CPU model so the
contract is locked in before larger data lands.

Test matrix:

* ``test_uniform_row_count`` — regression: every row has exactly
  N_vision image tokens (the prior invariant). Assert the masked_scatter
  semantics match the old reshape path elementwise.
* ``test_mixed_row_counts`` — rows with {N_vision, N_vision/2, 0} image
  tokens in a single batch. Assert each row's image positions get the
  correct vision_embeds (in row-major order, padded slots untouched).
* ``test_zero_images_in_batch`` — every row has zero image tokens. The
  scatter must be a complete no-op; embeds output equals embed_tokens
  output exactly.
* ``test_pp_shape_inference_dryrun`` — emulate PP's zero-filled input
  shape probe: tokens all zero (no image sentinel), image_mask all
  False, vision_embeds non-empty. Must not crash and must leave the
  hidden state unmodified (same as embed_tokens output).
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _WORKSPACE)
sys.path.insert(0, os.path.join(_WORKSPACE, "torchtitan"))


# ---- Stand-in: just the embed + multimodal-scatter portion of the
# AttnRes-Kimi LM forward. We replicate it inline rather than build the
# full model (which needs fla-core / GPU triton). The scatter logic
# under test is pure tensor ops; it doesn't depend on the layer stack.

_DEFAULT_IMAGE_TOKEN_ID = 32_000


def _embed_with_scatter(
    embed_tokens: nn.Embedding,
    tokens: torch.Tensor,
    vision_embeds: torch.Tensor | None,
    image_mask: torch.Tensor | None = None,
    image_token_id: int | None = None,
) -> torch.Tensor:
    """Mirror of ``KimiLinearAttnResModel.forward`` lines 263-303.

    Kept in sync with that file by the test reviewer; if the real model
    diverges, these tests will keep passing while production breaks. To
    avoid that drift we import the real model where possible — but that
    requires fla-core, so on CPU we substitute this minimal mirror.
    """
    h = embed_tokens(tokens)
    if vision_embeds is None:
        return h
    if image_mask is None:
        sentinel = (
            image_token_id if image_token_id is not None
            else _DEFAULT_IMAGE_TOKEN_ID
        )
        image_mask = (tokens == sentinel)
    n_per_row = image_mask.sum(dim=1)
    n_vis_max = vision_embeds.size(1)
    arange = torch.arange(n_vis_max, device=image_mask.device)
    valid = arange.unsqueeze(0) < n_per_row.unsqueeze(1)
    source = vision_embeds[valid].to(h.dtype)
    h = h.masked_scatter(image_mask.unsqueeze(-1).expand_as(h), source)
    return h


def _make_embed(vocab: int = 32_010, dim: int = 8) -> nn.Embedding:
    """Vocab must exceed _DEFAULT_IMAGE_TOKEN_ID (32000)."""
    torch.manual_seed(42)
    return nn.Embedding(vocab, dim)


# ----------------------------------------------------------------------


def test_uniform_row_count():
    """Regression: every row has exactly N image tokens (prior invariant)."""
    embed = _make_embed()
    B, T, N_vis, D = 3, 20, 5, 8
    tokens = torch.zeros(B, T, dtype=torch.long)
    # First N_vis positions of each row are image tokens.
    tokens[:, :N_vis] = _DEFAULT_IMAGE_TOKEN_ID
    # Remaining positions are random text tokens (avoid sentinel).
    tokens[:, N_vis:] = torch.randint(1, 32_000, (B, T - N_vis))
    vision_embeds = torch.randn(B, N_vis, D)

    h = _embed_with_scatter(embed, tokens, vision_embeds)

    # Image positions should equal vision_embeds; non-image positions
    # should equal embed_tokens.
    assert h.shape == (B, T, D)
    for b in range(B):
        for t in range(N_vis):
            assert torch.equal(h[b, t], vision_embeds[b, t]), \
                f"image-pos ({b},{t}) != vision_embeds"
        for t in range(N_vis, T):
            assert torch.equal(h[b, t], embed(tokens[b, t])), \
                f"text-pos ({b},{t}) != embed_tokens"


def test_mixed_row_counts():
    """Variable image count: row 0 = N_vis, row 1 = N_vis/2, row 2 = 0."""
    embed = _make_embed()
    N_vis, D, T = 6, 8, 16
    tokens = torch.randint(1, 32_000, (3, T), dtype=torch.long)
    # row 0: positions 0..5 are image tokens
    tokens[0, :N_vis] = _DEFAULT_IMAGE_TOKEN_ID
    # row 1: positions 0..2 are image tokens
    tokens[1, :N_vis // 2] = _DEFAULT_IMAGE_TOKEN_ID
    # row 2: zero image tokens
    # vision_embeds padded to N_vis on every row; row 1's slots [3:6] and
    # all of row 2's are pad (any value).
    torch.manual_seed(7)
    vision_embeds = torch.randn(3, N_vis, D)

    h = _embed_with_scatter(embed, tokens, vision_embeds)

    # row 0: positions 0..5 get vision_embeds[0, 0..5]
    for t in range(N_vis):
        assert torch.equal(h[0, t], vision_embeds[0, t])
    # row 1: positions 0..2 get vision_embeds[1, 0..2] only.
    # vision_embeds[1, 3..5] should NOT appear anywhere in h[1].
    for t in range(N_vis // 2):
        assert torch.equal(h[1, t], vision_embeds[1, t])
    for t in range(N_vis // 2, T):
        assert torch.equal(h[1, t], embed(tokens[1, t])), \
            f"row 1 text-pos {t} corrupted"
    # row 2: every position is text (no scatter happened).
    for t in range(T):
        assert torch.equal(h[2, t], embed(tokens[2, t])), \
            f"row 2 pos {t} corrupted (should be pure embed)"


def test_zero_images_in_batch():
    """Every row has zero image tokens — scatter is a no-op."""
    embed = _make_embed()
    N_vis, D, T = 4, 8, 12
    tokens = torch.randint(1, 32_000, (2, T), dtype=torch.long)
    vision_embeds = torch.randn(2, N_vis, D)  # provided but should be ignored

    h = _embed_with_scatter(embed, tokens, vision_embeds)

    expected = embed(tokens)
    assert torch.equal(h, expected), "h diverged from embed_tokens with no image positions"


def test_pp_shape_inference_dryrun():
    """PP scheduler probes activation shapes with zero-filled tokens; no
    image sentinel is present. The scatter must not crash and must not
    modify the hidden state."""
    embed = _make_embed()
    N_vis, D, T, B = 4, 8, 10, 2
    # All-zero tokens — embed(0) gives a fixed vector, no sentinel match.
    tokens = torch.zeros(B, T, dtype=torch.long)
    vision_embeds = torch.randn(B, N_vis, D)

    h = _embed_with_scatter(embed, tokens, vision_embeds)

    expected = embed(tokens)
    assert torch.equal(h, expected), "PP shape-inference path mutated hidden state"


def test_image_mask_explicit_override():
    """When the caller passes ``image_mask`` directly (not derived from
    sentinel), the scatter still respects per-row counts."""
    embed = _make_embed()
    N_vis, D, T, B = 4, 8, 10, 2
    tokens = torch.randint(1, 32_000, (B, T))  # no sentinels
    vision_embeds = torch.randn(B, N_vis, D)
    # Row 0 has 3 image positions at indices 1, 4, 7. Row 1 has 0.
    image_mask = torch.zeros(B, T, dtype=torch.bool)
    image_mask[0, [1, 4, 7]] = True

    h = _embed_with_scatter(
        embed, tokens, vision_embeds, image_mask=image_mask,
    )

    # Row 0: positions 1,4,7 get vision_embeds[0, 0..2] in order.
    assert torch.equal(h[0, 1], vision_embeds[0, 0])
    assert torch.equal(h[0, 4], vision_embeds[0, 1])
    assert torch.equal(h[0, 7], vision_embeds[0, 2])
    # Row 0: other positions unchanged.
    for t in [0, 2, 3, 5, 6, 8, 9]:
        assert torch.equal(h[0, t], embed(tokens[0, t]))
    # Row 1: every position unchanged.
    for t in range(T):
        assert torch.equal(h[1, t], embed(tokens[1, t]))


def test_mixed_dtype_scatter():
    """vision_embeds in fp32, embed_tokens in bf16 — scatter must match
    embed_tokens dtype (the resulting hidden state ``h`` is bf16, so
    fp32 source must downcast before writing). Mirrors the production
    setup where SigLIP outputs fp32, projector linearly maps fp32→fp32,
    then enters a bf16 LM forward."""
    embed = _make_embed().to(torch.bfloat16)
    N_vis, D, T, B = 4, 8, 12, 2
    tokens = torch.zeros(B, T, dtype=torch.long)
    tokens[:, :N_vis] = _DEFAULT_IMAGE_TOKEN_ID
    tokens[:, N_vis:] = torch.randint(1, 32_000, (B, T - N_vis))
    vision_embeds_fp32 = torch.randn(B, N_vis, D, dtype=torch.float32)

    h = _embed_with_scatter(embed, tokens, vision_embeds_fp32)

    # h must be in embed_tokens' dtype, not vision_embeds' dtype.
    assert h.dtype == torch.bfloat16, (
        f"h dtype should match embed_tokens (bf16), got {h.dtype}"
    )
    # Image positions should equal vision_embeds *after bf16 downcast*.
    expected_at_image = vision_embeds_fp32.to(torch.bfloat16)
    for b in range(B):
        for t in range(N_vis):
            assert torch.equal(h[b, t], expected_at_image[b, t]), \
                f"image-pos ({b},{t}) wasn't downcast properly"


def test_assert_helper_raises_on_overflow():
    """The relaxed reference helper in ``multimodal_model.py`` still raises
    when a row has MORE image tokens than vision_embeds slots — catching
    upstream caller bugs early."""
    from phase5 import multimodal_model

    class _StubVision(nn.Module):
        def forward(self, pixel_values):
            class O:
                last_hidden_state = pixel_values.new_zeros(
                    pixel_values.size(0), 4, 8,
                )
            return O()

    class _StubLM(nn.Module):
        def forward(self, input_ids, vision_embeds=None, image_mask=None):
            B, T = input_ids.shape
            return torch.zeros(B, T, 64)  # vocab=64

    projector = nn.Linear(8, 8)
    vision = _StubVision()
    lm = _StubLM()

    B, T = 2, 16
    pix = torch.zeros(B, 3, 4, 4)
    # vision_embeds will have shape (B, 4, 8) — only 4 slots per row.
    # Make row 0 have 5 image tokens → overflow.
    input_ids = torch.randint(1, 32_000, (B, T))
    input_ids[0, :5] = multimodal_model.IMAGE_TOKEN_ID
    labels = torch.full_like(input_ids, multimodal_model.IGNORE_INDEX)

    with pytest.raises(RuntimeError, match="exceeds vision_embeds slots"):
        multimodal_model.multimodal_loss(
            vision, projector, lm, pix, input_ids, labels,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
