"""VQA-aligned OPD task — uses real VQA questions from mix665k conversations.

Diagnosis from Stage D-4 (2026-05-28):
  * Loss curve clean (0.59→0.56 over 50 steps with lr=1e-5)
  * GQA dropped 12.3% → 9.3% (BELOW baseline, NOT a regression bug)
  * Sample inspection: student outputs caption-style descriptions on
    GQA short-answer questions ("A woman in a red shirt and jeans..."
    when gold is "traffic light"). The student is being correctly
    distilled toward the OPD prompt distribution (LlavaOpdTask used a
    fixed "Describe the image briefly..." prompt) but eval is GQA
    VQA short answer — different task format.

This task uses **real LLaVA-Instruct multi-turn VQA questions from
mix665k** as the OPD prompt. Each mix665k record has multi-turn
conversations: human (with <image>) / gpt / human / gpt / ... We
take the first human turn as the OPD question (stripped of <image>
placeholder), let the teacher generate the answer via its chat
template, distill student against teacher's response logits.

Why this should work:
  * mix665k VQA questions match the format student SFT was trained on
    (LLaVA-Instruct-150K is a subset of mix665k)
  * Teacher (Llama-3-LLaVA-NeXT-8B) is instruction-tuned on this
    exact data shape — it gives high-quality short answers
  * Task format aligned with GQA short-answer eval

Diversity:
  * 364K COCO records × ~3-5 distinct first-turn questions per record
    = effectively unbounded prompt pool for 600-step runs.
"""
from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass


# Strip the <image> placeholder + leading/trailing whitespace; mix665k
# inserts it at either start or end of the first human turn.
_IMG_RE = re.compile(r"\s*<image>\s*")


@dataclass
class VqaOpdRecord:
    prompt_text: str    # the user question (no <image>)
    image_path: str
    gold_caption: str = ""  # unused; kept for runner-API compatibility


class VqaAlignedOpdTask:
    """OPD task built from real VQA questions in mix665k conversations.

    Args:
        json_path: mix665k JSON.
        images_dir: dir containing ``coco/train2017/<file>.jpg``.
        seed: RNG seed.
        max_records: optional cap.
    """

    def __init__(
        self,
        json_path: str,
        images_dir: str,
        *,
        seed: int = 0,
        max_records: int | None = None,
    ):
        self.json_path = json_path
        self.images_dir = images_dir
        self._rng = random.Random(seed)

        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"mix665k JSON not found at {json_path}"
            )
        with open(json_path) as f:
            data = json.load(f)

        self._records: list[VqaOpdRecord] = []
        for entry in data:
            img = entry.get("image", "")
            if not img.startswith("coco/"):
                continue
            full_img = os.path.join(images_dir, img)
            if not os.path.isfile(full_img):
                continue
            convs = entry.get("conversations", [])
            # First human turn = the OPD question.
            user_q = None
            for c in convs:
                if c.get("from") == "human":
                    user_q = c.get("value", "")
                    break
            if not user_q:
                continue
            # Strip the <image> placeholder; mix665k puts it at
            # start, end, or on its own line.
            user_q = _IMG_RE.sub(" ", user_q).strip()
            if not user_q:
                continue
            self._records.append(VqaOpdRecord(
                prompt_text=user_q, image_path=full_img,
            ))
            if max_records is not None and len(self._records) >= max_records:
                break

        if not self._records:
            raise RuntimeError(
                f"No VQA records resolved from {json_path}."
            )

    def __len__(self) -> int:
        return len(self._records)

    def create_question(self) -> VqaOpdRecord:
        return self._rng.choice(self._records)

    def get_system_prompt(self) -> str:
        # LLaVA-style brief assistant framing; matches the SFT prompt
        # template and biases the student toward short answers.
        return (
            "You are a helpful vision assistant. Answer the question "
            "about the image concisely."
        )

    @staticmethod
    def reward_function(episode) -> float:
        # OPD has no reward — teacher logits are supervision.
        return 0.0


def _smoke():
    task = VqaAlignedOpdTask(
        json_path="/workspace/llava_opd/llava_v1_5_mix665k.json",
        images_dir="/workspace/llava_opd/images",
        max_records=200,
    )
    print(f"[vqa-opd] loaded {len(task)} VQA records (capped 200)")
    for _ in range(3):
        r = task.create_question()
        print(f"  Q: {r.prompt_text[:120]}")
        print(f"  img: {r.image_path}")
        assert os.path.isfile(r.image_path)
    print("[vqa-opd] SMOKE PASSED")


if __name__ == "__main__":
    _smoke()
