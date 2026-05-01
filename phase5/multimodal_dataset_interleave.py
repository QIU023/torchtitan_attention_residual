"""Image-text interleave dataset wrapper (phase 6 task B2).

LLaVA-Pretrain emits records where image tokens are strictly contiguous
at the prefix:

    [<img> × 196] [BOS] [caption tokens] [EOS]

InternVL / DeepSeek-VL2 / Kimi-VL 1.5 use *interleaved* layouts where
image tokens can appear at arbitrary positions inside the text:

    [BOS] [text...] [<img> × 196] [more text...] [EOS]
    [BOS] [text A...] [<img> × 196] [text B...] [<img> × 196] [text C...] [EOS]

The model side already supports this — phase 6 task B1's
``test_image_mask_explicit_override`` proves the masked_scatter path
correctly writes vision_embeds to image_mask positions regardless of
prefix vs non-prefix. What was missing: a dataset that actually emits
non-prefix layouts.

This module provides ``InterleavedLlavaPretrainDataset`` — a wrapper
that inherits from ``LlavaPretrainDataset`` and re-arranges the
emitted ``input_ids`` / ``labels`` to place the image-token block
inside the caption (or before, after, or split into two halves).

Modes:

* ``"prefix"`` — original LLaVA layout (``[<img> × N] [BOS] [caption]``).
  The default behavior; this is what ``LlavaPretrainDataset`` emits.
* ``"interior"`` — split the caption in half, put the image block
  between the halves: ``[BOS] [caption_half_1] [<img> × N] [caption_half_2] [EOS]``.
* ``"random"`` — pick one of {prefix, interior} per-record uniformly. Useful
  for stress-testing the model + collate under variable layouts.

The number of image tokens per row remains exactly ``N_VISION_TOKENS``
(196). Total sequence length stays at ``GLOBAL_SEQ_LEN_DEFAULT`` (258)
after padding. So neither the model's variable-image-count path nor
the collate's PP-shape-stability invariant is exercised — this is
purely a layout test.

Usage::

    from phase5.multimodal_dataset_interleave import InterleavedLlavaPretrainDataset
    ds = InterleavedLlavaPretrainDataset(
        json_path="...",
        images_dir="...",
        tokenizer=tok,
        image_processor=proc,
        dp_rank=0, dp_world_size=4,
        layout="random",
    )

The trainer wiring (replacing the stock ``LlavaPretrainDataset`` with
this wrapper) is a separate flag-driven follow-up.
"""
from __future__ import annotations

import random
from typing import Iterator

import torch

from phase5.multimodal_dataset import (
    IGNORE_INDEX,
    IMAGE_TOKEN_ID,
    N_VISION_TOKENS,
    LlavaPretrainDataset,
)


VALID_LAYOUTS = ("prefix", "interior", "random")


class InterleavedLlavaPretrainDataset(LlavaPretrainDataset):
    """Variant of ``LlavaPretrainDataset`` that re-arranges the emitted
    ``input_ids`` / ``labels`` according to a layout policy.

    Args:
        layout: One of ``VALID_LAYOUTS``. ``"random"`` uses a
            per-record RNG seeded by ``dp_rank + dp_world_size *
            record_index`` so layout assignment is reproducible across
            ranks for the same seed.
        layout_seed: Base seed for the per-record layout RNG.
    """

    def __init__(
        self, *args, layout: str = "interior", layout_seed: int = 0, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if layout not in VALID_LAYOUTS:
            raise ValueError(
                f"layout must be one of {VALID_LAYOUTS}; got {layout!r}"
            )
        self._layout = layout
        self._layout_seed = layout_seed

    def _rearrange(self, prefix_input_ids: list[int],
                   prefix_labels: list[int],
                   record_idx: int) -> tuple[list[int], list[int]]:
        """Take a prefix-layout (input_ids, labels) and return a
        rearranged version per ``self._layout``.

        Both inputs are lists of equal length. The image block is
        identified as the leading ``N_VISION_TOKENS`` positions
        (LlavaPretrainDataset always emits this layout).
        """
        # Detect the leading image-token block dynamically. The dataset
        # always emits image tokens contiguously at the start; count
        # how many.
        n_img = 0
        for t in prefix_input_ids:
            if t == IMAGE_TOKEN_ID:
                n_img += 1
            else:
                break
        # If no image tokens or no text following, return as-is.
        if n_img == 0 or len(prefix_input_ids) <= n_img:
            return prefix_input_ids, prefix_labels

        img_block_input = prefix_input_ids[:n_img]
        img_block_label = prefix_labels[:n_img]
        text_input = prefix_input_ids[n_img:]
        text_label = prefix_labels[n_img:]

        layout = self._layout
        if layout == "random":
            rng = random.Random(
                self._layout_seed + self.dp_world_size * record_idx + self.dp_rank
            )
            layout = rng.choice(("prefix", "interior"))

        if layout == "prefix":
            # No-op (matches the parent dataset's emission).
            return prefix_input_ids, prefix_labels

        if layout == "interior":
            # Split text_input in half (after BOS), insert image block.
            # text_input layout: [bos, c0, c1, ..., cN, eos]
            if len(text_input) < 4:
                return prefix_input_ids, prefix_labels
            mid = 1 + (len(text_input) - 2) // 2  # split point inside the caption
            new_input = (
                text_input[:mid] + img_block_input + text_input[mid:]
            )
            new_label = (
                text_label[:mid] + img_block_label + text_label[mid:]
            )
            return new_input, new_label

        raise RuntimeError(f"unhandled layout: {layout}")  # unreachable

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        # Wrap parent iterator: for each emitted record, reshape input_ids
        # and labels per the layout policy.
        for idx, record in enumerate(super().__iter__()):
            ii = record["input_ids"].tolist()
            ll = record["labels"].tolist()
            new_input, new_label = self._rearrange(ii, ll, idx)
            yield {
                "pixel_values": record["pixel_values"],
                "input_ids": torch.tensor(new_input, dtype=torch.long),
                "labels": torch.tensor(new_label, dtype=torch.long),
            }
