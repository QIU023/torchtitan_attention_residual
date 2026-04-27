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
        max_caption_tokens: int = 64,
    ):
        self.json_path = json_path
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.max_caption_tokens = max_caption_tokens

        if not os.path.isfile(json_path):
            raise FileNotFoundError(json_path)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(self.images_dir)

        # Load full record list once (it's ~558K records, ~100 MB JSON).
        with open(json_path, "r") as f:
            self.records = json.load(f)

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
            for idx in range(my_offset, len(self.records), total_stride):
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

    # Stateful protocol — minimal no-op for continued pretraining
    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        pass


def collate_with_pad(batch, pad_id: int = 0):
    """Pad input_ids / labels to max length in batch; stack pixel_values.

    Returns ``(input_dict, labels)`` matching torchtitan's
    ``batch_generator`` contract (``input_dict, labels = batch``).
    `pixel_values` and `input_ids` go in the input_dict; `labels`
    is returned separately so torchtitan's IGNORE-index counting in
    `train_step` works without modification.
    """
    max_len = max(b["input_ids"].size(0) for b in batch)
    B = len(batch)
    pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
    input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
    labels = torch.full((B, max_len), IGNORE_INDEX, dtype=torch.long)
    for i, b in enumerate(batch):
        L = b["input_ids"].size(0)
        input_ids[i, :L] = b["input_ids"]
        labels[i, :L] = b["labels"]
    input_dict = {
        "pixel_values": pixel_values,
        "input": input_ids,   # key name "input" for torchtitan trainer compat
    }
    return input_dict, labels
