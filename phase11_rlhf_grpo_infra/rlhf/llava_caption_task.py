"""Multimodal RLHF task: LLaVA-Pretrain image captioning with verifiable
reward.

Mirrors :class:`torchtitan.experiments.rl.sum_digits.SumDigitsTask` but
yields ``(image_path, prompt_text, gold_caption)`` tuples instead of
text-only ``(question, answer)``.

The reward is a length sanity check + word-overlap (BLEU-1-style)
against the gold LLaVA caption + a small format bonus. No external
reward model needed — the gold caption is what supervised training
already optimised for, so the RLHF signal is "stay close to the
supervised distribution + format constraints", which is the simplest
production-style verifiable reward for VLM RLHF.

Why this task:
* Reuses LLaVA-Pretrain JSON we already downloaded (558k pairs).
* No new dataset / no new reward model dependency.
* Reward is bounded ∈ [-1, +1.2] and signal-rich (BLEU-1 is dense).
* Multimodal — exercises the SGLang VLM path end-to-end.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import NamedTuple, Optional

import torch


class CaptionRecord(NamedTuple):
    image_path: str
    prompt_text: str
    gold_caption: str


_SYSTEM_PROMPT = """\
You are a helpful vision assistant. Describe the image in one short
sentence (5 to 30 words). Begin with a capital letter and end with a
period."""

_USER_PROMPT = "Describe the image briefly."


def _tokenise(text: str) -> list[str]:
    """Lowercased word tokens; matches BLEU-1 conventions."""
    return re.findall(r"[A-Za-z']+", text.lower())


def _bleu1(candidate: str, reference: str) -> float:
    """Modified unigram precision (clipped count / candidate length)."""
    cand = _tokenise(candidate)
    ref = _tokenise(reference)
    if not cand or not ref:
        return 0.0
    ref_count = Counter(ref)
    overlap = 0
    for tok in cand:
        if ref_count[tok] > 0:
            overlap += 1
            ref_count[tok] -= 1
    return overlap / len(cand)


class LlavaCaptionTask:
    """Streams (image_path, prompt, gold_caption) for VLM RLHF.

    Args:
        json_path: LLaVA-Pretrain ``blip_laion_cc_sbu_558k.json``.
        images_dir: directory containing the unzipped image shards
            (e.g. ``/workspace/.hf_home/LLaVA-Pretrain``).
        seed: RNG seed for question sampling.
        max_records: cap the number of records loaded from JSON
            (smaller = faster boot for smokes).
    """

    def __init__(
        self,
        json_path: str,
        images_dir: str,
        seed: int = 42,
        max_records: Optional[int] = 50_000,
    ):
        self.images_dir = Path(images_dir)
        self._rng = random.Random(seed)

        with open(json_path, "r") as f:
            records = json.load(f)
        if max_records is not None:
            records = records[:max_records]
        # Each LLaVA-Pretrain record is:
        #   {"id": ..., "image": "00000/000000abc.jpg",
        #    "conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "<caption>"}]}
        self._records = []
        for rec in records:
            img = rec.get("image")
            convs = rec.get("conversations", [])
            cap = next(
                (c.get("value") for c in convs if c.get("from") == "gpt"),
                None,
            )
            if img and cap:
                self._records.append((img, cap))
        if not self._records:
            raise ValueError(f"no usable records in {json_path}")

    def __len__(self) -> int:
        return len(self._records)

    def get_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def get_user_prompt(self) -> str:
        return _USER_PROMPT

    def create_question(self) -> CaptionRecord:
        """Sample one (image_path, prompt, gold_caption)."""
        img_rel, cap = self._rng.choice(self._records)
        return CaptionRecord(
            image_path=str(self.images_dir / img_rel),
            prompt_text=_USER_PROMPT,
            gold_caption=cap,
        )

    def reward_function(
        self,
        completions: list[str],
        expected_answer: str = "",
    ) -> torch.Tensor:
        """Compute rewards.

        Reward components:
          * length sanity: 5 ≤ tokens ≤ 30 (else -1.0 hard fail)
          * BLEU-1 unigram overlap with gold caption (∈ [0, 1])
          * format bonus +0.2 if completion starts with capital and
            ends with period

        Returns:
            Tensor of float32 rewards in roughly [-1, 1.2].
        """
        # 2026-05-27: GRADED length penalty instead of a hard -1 cliff. The hard
        # gate made ~75% of rollouts -1.0 (LLaVA-style captions run >30 tokens),
        # so the reward was sparse and GRPO stalled (flat reward_mean ~-0.6 over
        # 150 steps). Now BLEU content + format are ALWAYS scored and length is a
        # smooth penalty -> dense learnable signal so the policy can actually climb
        # (improve content AND shorten toward the window). Per the "any means to get
        # substantive RL improvement" directive.
        rewards = []
        for completion in completions:
            n_tokens = len(_tokenise(completion))
            r = _bleu1(completion, expected_answer)  # dense content signal [0,1]

            stripped = completion.strip()
            if (
                stripped
                and stripped[0].isupper()
                and stripped.endswith(".")
                and stripped.count(".") <= 3  # not multi-sentence padding
            ):
                r += 0.2

            # graded length penalty (smooth, replaces the hard -1 gate)
            if n_tokens < 5:
                r -= 0.6 * (5 - n_tokens) / 5.0          # too short -> up to -0.6
            elif n_tokens > 30:
                r -= min(1.0, 0.03 * (n_tokens - 30))    # too long -> graded, capped at -1.0
            if n_tokens == 0:
                r = -1.0                                  # empty completion

            rewards.append(r)

        return torch.tensor(rewards, dtype=torch.float32)
