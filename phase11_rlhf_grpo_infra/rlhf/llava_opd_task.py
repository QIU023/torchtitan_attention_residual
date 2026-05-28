"""OPD prompt pool from llava_v1_5_mix665k.json COCO entries.

GRPO needs a reward fn + gold answer per record. OPD does not — the
teacher's logits ARE the supervision. So this task surfaces (prompt,
image) pairs and stubs reward/gold to keep the existing runner's
Episode shape unchanged.

COCO is the cleanest LLaVA subset for distillation:
  - 118K real natural images (no OCR / chart corner cases).
  - 5 caption-style human references per image (only used as the
    student-side prompt seed; teacher provides actual logits).
  - Already on disk at /workspace/llava_opd/images/coco/train2017/.

Filter rule: keep only entries whose ``image`` field starts with
``coco/`` (skips GQA/OCR-VQA/TextVQA/VG subdirs that ship in mix665k
but require separate downloads).
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Iterable


@dataclass
class OpdRecord:
    """The launcher feeds these into the generator + (for OPD) the trainer.

    ``gold_caption`` is intentionally empty — OPD does not score against
    a single reference; the teacher's logits define the supervision.
    The field is preserved so the runner's Episode-construction code
    can stay shared between GRPO and OPD paths.
    """
    prompt_text: str
    image_path: str
    gold_caption: str = ""


class LlavaOpdTask:
    """OPD task over mix665k COCO entries.

    Args:
        json_path: path to ``llava_v1_5_mix665k.json``.
        images_dir: directory containing ``coco/train2017/<file>.jpg``.
        prompt: instruction to send to the student. Default mirrors
            the LLaVA-Instruct "describe the image" style. Distillation
            target doesn't depend on the prompt shape — any reasonable
            instruction works.
        seed: RNG seed for record selection.
        max_records: optional cap (None = use all matching).
    """

    DEFAULT_PROMPT = (
        "Describe the image in one or two short sentences, focusing on "
        "the most salient objects and what they are doing."
    )

    def __init__(
        self,
        json_path: str,
        images_dir: str,
        *,
        prompt: str | None = None,
        seed: int = 0,
        max_records: int | None = None,
    ):
        self.json_path = json_path
        self.images_dir = images_dir
        self.prompt = prompt or self.DEFAULT_PROMPT
        self._rng = random.Random(seed)

        if not os.path.isfile(json_path):
            raise FileNotFoundError(
                f"llava_v1_5_mix665k.json not found at {json_path}; "
                "download via the mix665k recipe first."
            )
        with open(json_path) as f:
            data = json.load(f)

        # Keep only entries with a COCO image, and where that file
        # actually exists on disk (skips entries whose image set
        # we did NOT download).
        self._records: list[OpdRecord] = []
        for entry in data:
            img = entry.get("image", "")
            if not img.startswith("coco/"):
                continue
            full = os.path.join(images_dir, img)
            if not os.path.isfile(full):
                continue
            self._records.append(OpdRecord(
                prompt_text=self.prompt,
                image_path=full,
            ))
            if max_records is not None and len(self._records) >= max_records:
                break

        if not self._records:
            raise RuntimeError(
                f"No COCO records resolved from {json_path} with images "
                f"under {images_dir}. Check ``{images_dir}/coco/train2017/``."
            )

    def __len__(self) -> int:
        return len(self._records)

    def create_question(self) -> OpdRecord:
        """Return a single random record (matches GqaVqaTask / LlavaCaptionTask
        API used by the runner main loop)."""
        return self._rng.choice(self._records)

    def get_system_prompt(self) -> str:
        """Match other tasks' API — no system framing needed for OPD."""
        return ""

    @staticmethod
    def reward_function(episode) -> float:
        """OPD has no reward — the teacher's logits are the supervision.

        Returned 0.0 keeps the existing grader / advantage code paths
        running as no-ops (mean=0, std=0 → advantage=0). OPDTrainer.step
        ignores both reward and advantage entirely; this stub only keeps
        the runner from crashing when it asks the grader for a score.
        """
        return 0.0


def _smoke():
    """Light smoke: load + count + sample one record + verify image exists."""
    task = LlavaOpdTask(
        json_path="/workspace/llava_opd/llava_v1_5_mix665k.json",
        images_dir="/workspace/llava_opd/images",
        max_records=100,
    )
    print(f"[opd-task] loaded {len(task)} COCO records (capped at 100)")
    r = task.create_question()
    print(f"[opd-task] sample: prompt={r.prompt_text[:60]}...")
    print(f"[opd-task]         image={r.image_path}")
    assert os.path.isfile(r.image_path), "sample image missing"
    print(f"[opd-task]         reward_fn(stub) = {LlavaOpdTask.reward_function(None)}")
    print("[opd-task] SMOKE PASSED")


if __name__ == "__main__":
    _smoke()
