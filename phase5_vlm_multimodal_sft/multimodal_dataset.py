"""LLaVA-Pretrain dataset → (pixel_values, input_ids, labels) batches.

Sequence layout per sample:
    [<img> × N_vision] [BOS] [caption tokens] [EOS]

Where:
* `<img>` is a placeholder token id (we use `IMAGE_TOKEN_ID = 32000`,
  one of Llama-3.1's reserved special tokens). At forward time the LM's
  embedding for that id is replaced by the projector's output.
* `labels` is `-100` at all image-token positions and at BOS, so loss
  is computed only on caption tokens + EOS.

N_vision = 196 (SigLIP-Base @ 224×224 patch16 → 14×14 patches).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterator

import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


IMAGE_TOKEN_ID = 32_000   # Llama-3.1 reserved special token, repurposed as <image>
N_VISION_TOKENS = 196     # SigLIP-Base patch16 224x224 → 14*14 = 196
IGNORE_INDEX = -100

# Fixed seq_len used by collate_with_pad_global. All microbatches must have
# IDENTICAL shape so PP's P2P recv buffers (sized at the first microbatch's
# shape) don't crash on later batches with different padded length.
#   N_VISION_TOKENS + bos + max_caption + eos = 196 + 1 + 60 + 1 = 258
GLOBAL_SEQ_LEN_DEFAULT = 258
MAX_CAPTION_TOKENS_DEFAULT = 60


class LlavaPretrainDataset(IterableDataset):
    """Streams LLaVA-Pretrain image-caption pairs.

    Each yield: dict with
        pixel_values:  Tensor [3, 224, 224] preprocessed for SigLIP
        input_ids:     LongTensor [N_vision + caption_len + 2]
        labels:        LongTensor [N_vision + caption_len + 2]
                       (IGNORE_INDEX at image + BOS positions)

    Sharded across (dp_rank, world_size). Loops infinitely.
    """

    def __init__(
        self,
        json_path: str,
        images_dir: str,
        tokenizer,
        image_processor,
        dp_rank: int,
        dp_world_size: int,
        max_caption_tokens: int = MAX_CAPTION_TOKENS_DEFAULT,
        split: str = "train",
        val_samples: int = 512,
        infinite: bool = True,
    ):
        """
        Args:
            split: "train" uses records[:-val_samples]; "val" uses the last
                ``val_samples`` records. The two index sets are disjoint and
                deterministic — the held-out val set is never seen by training.
            val_samples: Size of the held-out validation tail. If 0, the full
                dataset is used for "train" and "val" is empty.
            infinite: When True the iterator loops forever (training). When
                False it yields a single pass (validation), so a val pass
                terminates instead of streaming indefinitely.
        """
        self.json_path = json_path
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.max_caption_tokens = max_caption_tokens
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val'; got {split!r}")
        self.split = split
        self.val_samples = val_samples
        self.infinite = infinite

        if not os.path.isfile(json_path):
            raise FileNotFoundError(json_path)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(self.images_dir)

        # Load full record list once (it's ~558K records, ~100 MB JSON).
        with open(json_path, "r") as f:
            all_records = json.load(f)

        # Deterministic held-out split: training never sees the val tail.
        if val_samples > 0 and val_samples < len(all_records):
            if split == "train":
                self.records = all_records[:-val_samples]
            else:
                self.records = all_records[-val_samples:]
        else:
            # val_samples == 0 (or pathologically large): no held-out split.
            self.records = all_records if split == "train" else []

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        wi = get_worker_info()
        if wi is not None:
            local_stride = wi.num_workers
            local_offset = wi.id
        else:
            local_stride = 1
            local_offset = 0
        total_stride = self.dp_world_size * local_stride
        my_offset = self.dp_rank * local_stride + local_offset

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        if bos is None:
            bos = 128_000  # llama-3.1 BOS
        if eos is None:
            eos = 128_001

        while True:
            for idx in range(my_offset, len(self.records), total_stride):  # noqa: B007
                rec = self.records[idx]
                # LLaVA-Pretrain record format:
                #   {"id": "...", "image": "00000/000000000.jpg",
                #    "conversations": [{"from": "human", "value": "<image>\n..."},
                #                      {"from": "gpt", "value": "the caption text"}]}
                img_path = self.images_dir / rec["image"]
                if not img_path.is_file():
                    continue

                # Caption is always the gpt turn for LLaVA-Pretrain (single turn).
                caption = ""
                for turn in rec.get("conversations", []):
                    if turn.get("from") == "gpt":
                        caption = turn.get("value", "")
                        break
                if not caption.strip():
                    continue

                # Load + preprocess image
                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                pix = self.image_processor(images=image, return_tensors="pt")
                pixel_values = pix["pixel_values"][0]  # (3, H, W)

                # Tokenize caption (no BOS/EOS, we add them explicitly)
                caption_ids = self.tokenizer.encode(
                    caption, add_special_tokens=False,
                )[: self.max_caption_tokens]

                # Full sequence:
                #   [<img>] * N_vision  +  [bos]  +  caption  +  [eos]
                full = (
                    [IMAGE_TOKEN_ID] * N_VISION_TOKENS
                    + [bos]
                    + caption_ids
                    + [eos]
                )
                # Next-token prediction: input = full[:-1], labels = full[1:].
                # Then mask label positions whose target token is an image
                # token or BOS (we only train on predicting caption + EOS).
                input_ids = full[:-1]
                labels = list(full[1:])
                for i, t in enumerate(labels):
                    if t == IMAGE_TOKEN_ID or t == bos:
                        labels[i] = IGNORE_INDEX

                yield {
                    "pixel_values": pixel_values,
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }

            # Validation dataloaders make a single pass and then stop, so the
            # val loop terminates. Training loops forever.
            if not self.infinite:
                break

    # Stateful protocol — minimal no-op for continued pretraining
    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


def collate_with_pad(batch, pad_id: int = 0,
                     global_seq_len: int = GLOBAL_SEQ_LEN_DEFAULT):
    """Pad to a FIXED `global_seq_len` so every microbatch has identical shape.

    Why fixed rather than per-batch max:
      Under PP, the scheduler pre-allocates P2P recv buffers from the FIRST
      microbatch's shape. A later microbatch padded to a different length
      crashes the receiving stage with a tensor-shape mismatch.
      Fixed-length pad eliminates this.

    Captions are dropped/truncated by the dataset's ``max_caption_tokens``
    so each yield is at most ``global_seq_len`` tokens. Rows shorter than
    ``global_seq_len`` are padded with ``pad_id``; their label positions
    are filled with ``IGNORE_INDEX`` so they don't contribute to loss.

    Returns ``(input_dict, labels)`` matching torchtitan's
    ``batch_generator`` contract (``input_dict, labels = batch``).
    """
    B = len(batch)
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    input_ids = torch.full((B, global_seq_len), pad_id, dtype=torch.long)
    labels = torch.full((B, global_seq_len), IGNORE_INDEX, dtype=torch.long)
    for i, b in enumerate(batch):
        L = min(b["input_ids"].size(0), global_seq_len)
        input_ids[i, :L] = b["input_ids"][:L]
        labels[i, :L] = b["labels"][:L]
    input_dict = {
        "pixel_values": pixel_values,
        "input": input_ids,   # key name "input" for torchtitan trainer compat
    }
    return input_dict, labels
