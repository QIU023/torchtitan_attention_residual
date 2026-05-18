"""LLaVA-Instruct-150K conversation-format dataset for SFT.

Differs from LLaVA-Pretrain (caption only) in:
* Multi-turn conversation: alternating human / gpt turns.
* Loss is supervised ONLY on gpt turns (human turns get IGNORE_INDEX).
* Image is COCO 2017 train (filename like "000000033471.jpg"),
  NOT the LLaVA-Pretrain set (different repo).

Sequence layout (single sample):
    [<img> × N_vision] [BOS] [Q1] [A1] [Q2] [A2] ... [EOS]
Where Q-tokens get IGNORE_INDEX in labels and A-tokens get their own
token IDs. Image + BOS positions also IGNORE_INDEX.

For 8-GPU 4D PP setup, all microbatches must have IDENTICAL shape.
We pad/truncate to ``GLOBAL_SFT_SEQ_LEN`` (default 384 tokens beyond
the 196 image tokens, total 384+196 = 580), which matches LLaVA-1.5
practice for 1024-token effective context.

Conversations exceeding the budget are truncated turn-by-turn from
the back: keep the earliest Q→A pairs that fit within budget.
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Iterator

import torch
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info


IMAGE_TOKEN_ID = 32_000
N_VISION_TOKENS = 196
IGNORE_INDEX = -100

# Total seq_len = N_VISION_TOKENS + GLOBAL_SFT_TEXT_LEN (incl BOS/EOS)
GLOBAL_SFT_TEXT_LEN_DEFAULT = 384
GLOBAL_SFT_SEQ_LEN_DEFAULT = N_VISION_TOKENS + GLOBAL_SFT_TEXT_LEN_DEFAULT


class LlavaInstructSFTDataset(IterableDataset):
    """Streams LLaVA-Instruct-150K conversation samples for SFT.

    Each yield: dict with
        pixel_values:  Tensor [3, 224, 224] preprocessed for SigLIP
        input_ids:     LongTensor [N_vision + text_len]
        labels:        LongTensor [N_vision + text_len]
                       (IGNORE_INDEX at image, BOS, all human-turn tokens;
                        real ids at gpt-turn tokens + EOS)

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
        text_len: int = GLOBAL_SFT_TEXT_LEN_DEFAULT,
        split: str = "train",
        val_samples: int = 0,
        infinite: bool = True,
        shuffle_seed: int = 0,
        stratified_val_per_source: int = 0,
    ):
        """Streams mix665k records, with optional held-out val split.

        Args:
            split: "train" or "val".
            val_samples: Size of the LEGACY tail-based val split (last N
                image-only records). DEPRECATED — mix665k has its data
                sources clustered contiguously in JSON order, so the tail
                is dominated by a single source (in mix665k, textvqa) and
                makes val a single-domain test, not representative of the
                mixed train distribution. Prefer ``stratified_val_per_source``.
                Kept for backward compatibility with prior runs.
            stratified_val_per_source: When >0, build val by sampling N
                records per data source (coco/gqa/vg/textvqa/...) with
                deterministic per-source seed. Train then excludes exactly
                those records. Total val size ≈ N × num_sources.
                Overrides ``val_samples`` when set.
            infinite: True for training (loops forever); False for val
                (single pass so the eval loop terminates).
            shuffle_seed: >0 and split=="train" shuffles train records with
                this seed. Val stays deterministic for cross-run comparison.
                Used to break deterministic-data crash loops (e.g. KDA
                assert at a fixed sample iteration index).
        """
        self.json_path = json_path
        self.images_dir = Path(images_dir)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.dp_rank = dp_rank
        self.dp_world_size = dp_world_size
        self.text_len = text_len
        self.seq_len = N_VISION_TOKENS + text_len
        self.split = split
        self.val_samples = val_samples
        self.infinite = infinite

        if not os.path.isfile(json_path):
            raise FileNotFoundError(json_path)
        if not self.images_dir.is_dir():
            raise FileNotFoundError(self.images_dir)

        with open(json_path, "r") as f:
            all_records = json.load(f)

        image_records = [r for r in all_records if r.get("image")]

        if stratified_val_per_source > 0:
            # Proper held-out val: sample N records per source so val mirrors
            # the train distribution. Sources are derived from the first path
            # segment of the image field (coco/gqa/vg/textvqa/ocr_vqa/...).
            # OCR-VQA records survive selection here but get silently filtered
            # at __iter__ time when img_path doesn't exist on disk (we keep
            # the records in val for accounting; iterator just skips them).
            by_source = {}
            for i, r in enumerate(image_records):
                img = r.get("image", "")
                src = img.split("/")[0] if "/" in img else "root"
                by_source.setdefault(src, []).append(i)
            val_indices: set[int] = set()
            for src in sorted(by_source.keys()):
                rng = random.Random(
                    0xC0FFEE ^ stratified_val_per_source ^ hash(src) & 0xFFFFFFFF
                )
                picks = rng.sample(
                    by_source[src],
                    min(stratified_val_per_source, len(by_source[src])),
                )
                val_indices.update(picks)
            if split == "train":
                self.records = [
                    r for i, r in enumerate(image_records) if i not in val_indices
                ]
            elif split == "val":
                self.records = [image_records[i] for i in sorted(val_indices)]
            else:
                raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        elif val_samples > 0:
            # LEGACY tail-based split. mix665k clusters records by source
            # contiguously in JSON; the tail is dominated by textvqa, so
            # this val is a single-source test, NOT representative.
            # Kept for backward compatibility with prior runs only.
            if split == "train":
                self.records = image_records[:-val_samples]
            elif split == "val":
                self.records = image_records[-val_samples:]
            else:
                raise ValueError(f"split must be 'train' or 'val', got {split!r}")
        else:
            self.records = all_records

        # Shuffle train split only (val stays deterministic for cross-run
        # comparison). Apply AFTER the split so val records are unchanged.
        if shuffle_seed > 0 and split == "train":
            random.Random(shuffle_seed).shuffle(self.records)

    def _tokenize_turn(self, role: str, text: str) -> list[int]:
        """Tokenize one turn with a small role marker. ``<image>`` tag
        is stripped (image tokens are pre-pended separately at the
        sequence level)."""
        text = text.replace("<image>", "").strip()
        # Mark turn boundary so the model can learn turn structure.
        # Llama-style "USER:" / "ASSISTANT:" — concise.
        prefix = "USER: " if role == "human" else "ASSISTANT: "
        full = prefix + text + ("\n" if role == "human" else "")
        return self.tokenizer.encode(full, add_special_tokens=False)

    def _build_text_tokens(
        self, conversations: list[dict], bos: int, eos: int,
    ) -> tuple[list[int], list[int]] | None:
        """Build (input_ids, labels) for the text portion only,
        excluding image tokens. Returns None if doesn't fit budget.
        """
        # text_len budget: BOS + content + EOS
        budget = self.text_len - 2
        token_ids: list[int] = [bos]
        labels: list[int] = [IGNORE_INDEX]
        for turn in conversations:
            role = turn.get("from", "")
            text = turn.get("value", "")
            if not text.strip() or role not in ("human", "gpt"):
                continue
            tokens = self._tokenize_turn(role, text)
            if len(token_ids) + len(tokens) > budget:
                # Truncate this turn to fit, then stop adding more
                space = max(0, budget - len(token_ids))
                if space < 8:
                    break
                tokens = tokens[:space]
            token_ids.extend(tokens)
            if role == "gpt":
                labels.extend(tokens)  # supervise gpt tokens
            else:
                labels.extend([IGNORE_INDEX] * len(tokens))
        # EOS at end (supervised so model learns to stop)
        if len(token_ids) >= self.text_len:
            token_ids = token_ids[: self.text_len - 1]
            labels = labels[: self.text_len - 1]
        token_ids.append(eos)
        labels.append(eos)

        if not any(l != IGNORE_INDEX for l in labels[:-1]):
            return None  # No gpt content survived → drop sample
        # Right-pad with eos / IGNORE_INDEX
        pad_n = self.text_len - len(token_ids)
        if pad_n > 0:
            token_ids.extend([eos] * pad_n)
            labels.extend([IGNORE_INDEX] * pad_n)
        return token_ids, labels

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

        bos = self.tokenizer.bos_token_id or 128_000
        eos = self.tokenizer.eos_token_id or 128_001

        while True:
            for idx in range(my_offset, len(self.records), total_stride):
                rec = self.records[idx]
                # mix665k has ~40K text-only records (no "image" key); skip.
                img_rel = rec.get("image")
                if not img_rel:
                    continue
                img_path = self.images_dir / img_rel
                if not img_path.is_file():
                    continue
                conversations = rec.get("conversations", [])
                if not conversations:
                    continue

                tok_pair = self._build_text_tokens(conversations, bos, eos)
                if tok_pair is None:
                    continue
                text_ids, text_labels = tok_pair

                try:
                    image = Image.open(img_path).convert("RGB")
                except Exception:
                    continue
                px = self.image_processor(
                    images=image, return_tensors="pt",
                )["pixel_values"][0]

                # Build full sequence then shift like LlavaPretrainDataset.
                # The trainer's loss_fn does NOT shift internally — it
                # computes CE(logits, labels) directly — so the dataset
                # must provide labels = input_ids[1:] (next-token target)
                # with IGNORE_INDEX wherever loss should be skipped.
                full_ids = (
                    [IMAGE_TOKEN_ID] * N_VISION_TOKENS
                    + text_ids
                )
                full_labels = (
                    [IGNORE_INDEX] * N_VISION_TOKENS
                    + text_labels
                )
                # Standard next-token shift:
                #   input_ids = full[:-1]  (positions 0..L-2)
                #   labels    = full[1:]   (the targets: position 1..L-1)
                input_ids = full_ids[:-1]
                labels = full_labels[1:]

                yield {
                    "pixel_values": px,
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            # Validation dataloaders make a single pass and stop, so the val
            # loop terminates. Training loops forever.
            if not self.infinite:
                break
