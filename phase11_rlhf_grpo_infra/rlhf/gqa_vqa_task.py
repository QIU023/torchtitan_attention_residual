"""Multimodal RLHF task: GQA visual question answering with a VERIFIABLE
exact-match reward — a real capability target (vs BLEU-on-pretrain-captions,
which is degenerate because the model already trained on LLaVA-Pretrain).

Same interface as LlavaCaptionTask (drop-in for run_grpo_llava_kimi.py):
  create_question() -> record with .image_path, .prompt_text, .gold_caption
  get_system_prompt(), get_user_prompt(), reward_function(completions, expected_answer)

The record field is named ``gold_caption`` (holding the GQA short answer) so the
runner's hardcoded ``r.gold_caption`` works unchanged.

Reward: normalized exact-match of the model's answer span against the GQA gold
short answer (lowercase, strip punctuation/articles). +1 correct / 0 wrong, with
an empty/rambling penalty. Verifiable, no reward model — the model can measurably
get better at answering, so reward_mean has real headroom to climb.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import NamedTuple

import torch


class VqaRecord(NamedTuple):
    image_path: str
    prompt_text: str
    gold_caption: str  # holds the GQA short answer (named for runner compat)


_SYSTEM_PROMPT = """\
You are a helpful vision assistant. Answer the question about the image with a
single short word or phrase."""

_USER_PROMPT = "Answer the question."

_ARTICLES = {"a", "an", "the"}


def _norm(text: str) -> str:
    """GQA-style normalization: lowercase, drop punctuation, drop articles."""
    toks = re.findall(r"[a-z0-9']+", text.lower())
    toks = [t for t in toks if t not in _ARTICLES]
    return " ".join(toks)


def _answer_span(completion: str) -> str:
    """The model's answer = first clause (up to . , ; newline). VLMs answer
    'No.' / 'A chair.' — take that, not the whole ramble."""
    head = re.split(r"[.,;\n]", completion.strip(), maxsplit=1)[0]
    return _norm(head)


class GqaVqaTask:
    def __init__(self, json_path: str, images_dir: str, max_records=None, seed: int = 0):
        self.images_dir = Path(images_dir)
        records = json.load(open(json_path))
        if max_records is not None:
            records = records[:max_records]
        self._records = []
        for r in records:
            img, q, a = r.get("image"), r.get("question"), r.get("answer")
            if img and q and a:
                self._records.append((img, q.strip(), a.strip().lower()))
        if not self._records:
            raise ValueError(f"no usable records in {json_path}")
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._records)

    def get_system_prompt(self) -> str:
        return _SYSTEM_PROMPT

    def get_user_prompt(self) -> str:
        return _USER_PROMPT

    def create_question(self) -> VqaRecord:
        img_rel, question, answer = self._rng.choice(self._records)
        return VqaRecord(
            image_path=str(self.images_dir / img_rel),
            prompt_text=question,          # the GQA question
            gold_caption=answer,           # the GQA short gold answer
        )

    def reward_function(self, completions: list[str], expected_answer: str = "") -> torch.Tensor:
        gold = _norm(expected_answer)
        rewards = []
        for c in completions:
            span = _answer_span(c)
            span_toks = span.split()
            # correct if the answer span equals gold, or starts with it, or gold
            # appears as a token in the (short) span.
            correct = bool(gold) and (
                span == gold
                or span.startswith(gold + " ")
                or (len(gold.split()) == 1 and gold in span_toks)
            )
            r = 1.0 if correct else 0.0
            n = len(_norm(c).split())
            if n == 0:
                r = -0.5                    # empty / non-answer
            elif n > 15:
                r -= 0.2                    # rambling penalty (encourage concise)
            rewards.append(r)
        return torch.tensor(rewards, dtype=torch.float32)
