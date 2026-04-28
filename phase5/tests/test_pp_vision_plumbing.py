"""Unit tests for the PP path's multimodal kwarg plumbing.

Bug 10.1 + 10.9 + 10.11 from HANDOFF_arm2_pp_adapter.md cluster around
"does ``vision_embeds`` survive the trip from the trainer through the
cache adapter into stage 0's wrapped model, while not corrupting middle
stages?". These tests use simple stand-ins for the wrapped model so the
plumbing is exercised without spinning up a 4-GPU process group.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

# Make the project workspace importable so phase5 / torchtitan resolve.
_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _WORKSPACE)
sys.path.insert(0, os.path.join(_WORKSPACE, "torchtitan"))

from torchtitan.experiments.attn_res.pipeline_adapter import (  # noqa: E402
    CrossStageCacheAdapter,
    _reset_rank_caches_for_testing,
)


class _RecordingFirstStage(nn.Module):
    """Stand-in for stage 0's wrapped ``KimiLinearAttnResModel``.

    Captures every (args, kwargs) tuple the adapter passes through and
    returns a fake ``(partial, blocks)`` tuple in the shape the adapter
    expects on a non-last stage.
    """

    def __init__(self, hidden: int = 8):
        super().__init__()
        self.hidden = hidden
        self.calls: list[tuple[tuple, dict]] = []
        self._return_only_new_blocks = False  # naive mode

    def forward(self, tokens: torch.Tensor, blocks=None, **kwargs):  # type: ignore[override]
        self.calls.append((tokens.shape, dict(kwargs)))
        B, T = tokens.shape
        partial = torch.zeros(B, T, self.hidden, requires_grad=False)
        # No new blocks committed on this stage in the test.
        new_blocks = partial.new_zeros((0, B, T, self.hidden))
        return partial, new_blocks


class _RecordingMiddleStage(nn.Module):
    """Stand-in for a middle stage. Receives ``(partial, blocks)`` positional."""

    def __init__(self):
        super().__init__()
        self.calls: list[tuple[tuple, dict]] = []
        self._return_only_new_blocks = False

    def forward(self, partial: torch.Tensor, blocks=None, **kwargs):  # type: ignore[override]
        self.calls.append((partial.shape, dict(kwargs)))
        B, T, D = partial.shape
        new_blocks = partial.new_zeros((0, B, T, D))
        return partial, new_blocks


@pytest.fixture(autouse=True)
def _reset_caches():
    _reset_rank_caches_for_testing()
    yield
    _reset_rank_caches_for_testing()


def test_first_stage_naive_mode_forwards_vision_embeds_kwarg():
    """Naive PP (no layout): adapter must pass ``vision_embeds`` kwarg
    through to the wrapped model on stage 0."""
    inner = _RecordingFirstStage(hidden=8)
    adapter = CrossStageCacheAdapter(
        inner, stage_id=0, num_stages=2,
        layout_tables=None,  # naive mode
    )
    tokens = torch.zeros(2, 16, dtype=torch.long)
    vision_embeds = torch.randn(2, 4, 8)

    partial, blocks = adapter(tokens, vision_embeds=vision_embeds)

    assert len(inner.calls) == 1
    args_shape, kwargs_seen = inner.calls[0]
    assert args_shape == (2, 16), f"input_ids shape lost: {args_shape}"
    assert "vision_embeds" in kwargs_seen, (
        f"adapter dropped vision_embeds; saw kwargs={list(kwargs_seen)}"
    )
    assert torch.allclose(kwargs_seen["vision_embeds"], vision_embeds)


def test_first_stage_naive_mode_passes_image_token_id_kwarg():
    """The ``image_token_id`` int kwarg (used to compute image_mask
    inside stage 0's forward when not explicitly supplied) must also
    survive the adapter dispatch."""
    inner = _RecordingFirstStage()
    adapter = CrossStageCacheAdapter(inner, stage_id=0, num_stages=2)
    tokens = torch.zeros(1, 4, dtype=torch.long)

    adapter(tokens, image_token_id=32_000)

    _, kwargs_seen = inner.calls[0]
    assert kwargs_seen.get("image_token_id") == 32_000


def test_middle_stage_naive_mode_passes_kwargs_through():
    """Middle stage receives (partial, blocks) positional + any kwargs
    that the schedule replicates (vision_embeds may be among them).
    Adapter must not refuse the extra kwargs even though the wrapped
    model on a middle stage will silently ignore them."""
    inner = _RecordingMiddleStage()
    adapter = CrossStageCacheAdapter(inner, stage_id=1, num_stages=2)
    partial_in = torch.randn(2, 16, 8)
    blocks_in = torch.randn(1, 2, 16, 8)
    vision_embeds = torch.randn(2, 4, 8)

    out = adapter(partial_in, blocks_in, vision_embeds=vision_embeds)

    assert len(inner.calls) == 1
    _, kwargs_seen = inner.calls[0]
    assert "vision_embeds" in kwargs_seen
    assert isinstance(out, tuple) and len(out) == 2


def test_collate_with_pad_global_seq_len_invariant():
    """Different per-row caption lengths → identical (B, GLOBAL_SEQ_LEN)
    output regardless of which row is longest. Required by PP P2P,
    where buffers are sized from the first microbatch."""
    from phase5.multimodal_dataset import (
        GLOBAL_SEQ_LEN_DEFAULT, IMAGE_TOKEN_ID, collate_with_pad,
    )

    short = {
        "pixel_values": torch.zeros(3, 224, 224),
        "input_ids": torch.tensor(
            [IMAGE_TOKEN_ID] * 196 + [128_000, 1, 128_001], dtype=torch.long
        ),
        "labels": torch.tensor(
            [-100] * 196 + [-100, 1, 128_001], dtype=torch.long
        ),
    }
    longer = {
        "pixel_values": torch.zeros(3, 224, 224),
        "input_ids": torch.tensor(
            [IMAGE_TOKEN_ID] * 196 + [128_000] + list(range(20)) + [128_001],
            dtype=torch.long,
        ),
        "labels": torch.tensor(
            [-100] * 196 + [-100] + list(range(20)) + [128_001],
            dtype=torch.long,
        ),
    }

    a, _ = collate_with_pad([short, longer], pad_id=0)
    b, _ = collate_with_pad([longer, short], pad_id=0)
    assert a["input"].shape == (2, GLOBAL_SEQ_LEN_DEFAULT)
    assert b["input"].shape == (2, GLOBAL_SEQ_LEN_DEFAULT)
    # Both microbatches' inputs/labels share the SAME shape across
    # different content orderings — this is the property PP P2P needs.
