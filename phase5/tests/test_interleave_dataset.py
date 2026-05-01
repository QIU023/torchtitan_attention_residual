"""Tests for the interleave dataset wrapper (phase 6 task B2).

The wrapper takes a prefix-layout LLaVA record:

    [<img> × 196] [BOS] [c0] [c1] ... [cN] [EOS]

and rearranges to one of:

* ``prefix``: original (no change)
* ``suffix``: [BOS] [c0..cN] [<img> × 196] [EOS]
* ``interior``: [BOS] [c0..cMID] [<img> × 196] [cMID..cN] [EOS]
* ``random``: per-record uniform pick

These tests exercise the rearrangement function in isolation without
needing the actual LLaVA-Pretrain dataset on disk; the dataset
construction is tested separately if/when the wrapper is wired into
the trainer.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _WORKSPACE)

from phase5.multimodal_dataset import IGNORE_INDEX, IMAGE_TOKEN_ID  # noqa: E402

# Don't actually import the dataset class (which requires LlavaPretrain
# JSON to instantiate); test the _rearrange method directly.
from phase5.multimodal_dataset_interleave import (  # noqa: E402
    VALID_LAYOUTS,
    InterleavedLlavaPretrainDataset,
)


# Override: build a stand-in that has _rearrange but doesn't call
# super().__init__ (which would need real LLaVA data on disk).
class _FakeInterleaved(InterleavedLlavaPretrainDataset):
    def __init__(self, layout="interior", layout_seed=0):
        # Skip parent __init__ entirely
        self._layout = layout
        self._layout_seed = layout_seed
        self.dp_world_size = 1
        self.dp_rank = 0


def _prefix_record(n_img=196, caption_tokens=(1, 2, 3, 4, 5)):
    """Build a prefix-layout (input_ids, labels) pair as
    LlavaPretrainDataset would emit."""
    bos = 128_000
    eos = 128_001
    full = (
        [IMAGE_TOKEN_ID] * n_img + [bos] + list(caption_tokens) + [eos]
    )
    input_ids = full[:-1]
    labels = list(full[1:])
    for i, t in enumerate(labels):
        if t == IMAGE_TOKEN_ID or t == bos:
            labels[i] = IGNORE_INDEX
    return input_ids, labels


def _img_count(input_ids):
    return sum(1 for t in input_ids if t == IMAGE_TOKEN_ID)


# ----------------------------------------------------------------------


def test_prefix_layout_is_noop():
    ds = _FakeInterleaved(layout="prefix")
    ii, ll = _prefix_record()
    out_i, out_l = ds._rearrange(ii, ll, record_idx=0)
    assert out_i == ii
    assert out_l == ll


def test_interior_layout_image_in_middle():
    ds = _FakeInterleaved(layout="interior")
    ii, ll = _prefix_record(n_img=4, caption_tokens=(11, 22, 33, 44, 55, 66))
    out_i, out_l = ds._rearrange(ii, ll, record_idx=0)
    assert _img_count(out_i) == 4
    # Image block is interior — neither at position 0 nor at last 4 positions
    img_positions = [i for i, t in enumerate(out_i) if t == IMAGE_TOKEN_ID]
    assert len(img_positions) == 4
    assert img_positions == list(range(img_positions[0], img_positions[0] + 4))
    # Image block is contiguous and not at the front
    assert img_positions[0] > 0
    # And not at the very end
    assert img_positions[-1] < len(out_i) - 1


def test_image_count_invariant_across_layouts():
    """Whatever layout, total image-token count == n_img."""
    for layout in ("prefix", "interior"):
        ds = _FakeInterleaved(layout=layout)
        ii, ll = _prefix_record(n_img=4, caption_tokens=(1, 2, 3, 4, 5))
        out_i, _ = ds._rearrange(ii, ll, record_idx=0)
        assert _img_count(out_i) == 4, f"layout={layout} dropped image tokens"


def test_total_length_invariant():
    """Rearrangement preserves total sequence length."""
    for layout in ("prefix", "interior"):
        ds = _FakeInterleaved(layout=layout)
        ii, ll = _prefix_record(n_img=4, caption_tokens=(1, 2, 3, 4, 5, 6))
        out_i, out_l = ds._rearrange(ii, ll, record_idx=0)
        assert len(out_i) == len(ii)
        assert len(out_l) == len(ll)


def test_random_layout_distribution():
    """Random layout selector picks each of {prefix,interior} at least
    once over 100 records (each per-record RNG uses record_idx)."""
    ds = _FakeInterleaved(layout="random")
    ii, ll = _prefix_record(n_img=4, caption_tokens=(1, 2, 3, 4, 5, 6))
    seen_layouts = set()
    for idx in range(100):
        out_i, _ = ds._rearrange(ii, ll, record_idx=idx)
        first_img = out_i.index(IMAGE_TOKEN_ID)
        if first_img == 0:
            seen_layouts.add("prefix")
        else:
            seen_layouts.add("interior")
    assert seen_layouts == {"prefix", "interior"}, (
        f"random layout didn't see both: {seen_layouts}"
    )


def test_valid_layouts_constant():
    assert "made_up" not in VALID_LAYOUTS
    assert set(VALID_LAYOUTS) == {"prefix", "interior", "random"}


def test_short_caption_falls_back_safely():
    """Edge case: caption is too short for interior split. Should fall
    back to prefix (no image-token loss)."""
    ds = _FakeInterleaved(layout="interior")
    # Caption of length 1 → text_input has [bos, c, eos] = 3 tokens.
    # interior requires >= 4 → fall back to prefix.
    ii, ll = _prefix_record(n_img=4, caption_tokens=(99,))
    out_i, _ = ds._rearrange(ii, ll, record_idx=0)
    assert out_i == ii  # no change
    assert _img_count(out_i) == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
