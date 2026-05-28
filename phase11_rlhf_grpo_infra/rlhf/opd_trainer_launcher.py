"""Launcher-side OPDTrainer adapter.

The upstream-bound ``OPDTrainer`` in
``torchtitan/experiments/rl/actors/opd_trainer.py`` is intentionally HF-
free: it takes the teacher / loss fn / tokenizer via inject endpoints
that accept *callables*. That works when the objects are pickleable and
already live in the trainer process — for unit tests, smaller stand-in
teachers, etc.

For our real Stage-C launcher the teacher is a full HF
``LlavaNextForConditionalGeneration`` (8B params, ~16 GB) which:
  * is not safely pickleable across an actor boundary, and
  * needs to be loaded *inside* the trainer's subprocess on its own
    GPU (so weights live next to where the loss + backward happen).

``LauncherOPDTrainer`` is a thin subclass that adds one extra endpoint
(``init_opd_components``) which takes only strings/ints (HF model id,
tokenizer path, device str) and constructs the heavy objects *inside*
the actor. The upstream base trainer doesn't need to know any of this.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import torch
from monarch.actor import endpoint

from torchtitan.experiments.rl.actors.opd_trainer import OPDTrainer

logger = logging.getLogger(__name__)


class LauncherOPDTrainer(OPDTrainer):
    """OPDTrainer that lazy-loads HF teacher + tokenizer + loss fn inside
    the actor process, given only string config.

    All HF / TRL imports happen inside ``init_opd_components`` so the
    main launcher process (which spawns the actor) doesn't pay for them.
    """

    @endpoint
    async def init_opd_components(
        self,
        teacher_model_id: str,
        teacher_device: str,
        tokenizer_path: str,
        opd_loss_module_dir: str,
        teacher_max_memory: dict | None = None,
    ) -> dict:
        """Build heavy components inside this trainer's process.

        Args:
            teacher_model_id: HF id for the teacher (e.g.
                ``llava-hf/llama3-llava-next-8b-hf``).
            teacher_device: CUDA device for the teacher when single-GPU
                (e.g. ``cuda:0``). Ignored when ``teacher_max_memory``
                is provided.
            teacher_max_memory: optional accelerate ``max_memory`` dict
                (``{device_idx: "10GiB", ...}``). When set, HF
                ``device_map="auto"`` spreads teacher layers across
                the keyed devices. Used by the runner to put the
                teacher on otherwise-idle GPUs (typically cuda:5-7
                physical → cuda:1-3 logical after the trainer
                bootstrap remaps ``CUDA_VISIBLE_DEVICES``).
            tokenizer_path: HF model dir whose tokenizer matches the
                student's vocab (used to decode Episode prompt/response
                token ids back to text for the teacher's input).
            opd_loss_module_dir: directory containing ``opd_loss.py``
                (the launcher-side adapter that calls
                ``trl.experimental.gkd.GKDTrainer.generalized_jsd_loss``).
                Added to ``sys.path`` so the import works regardless of
                the actor's working dir.

        Returns:
            Diagnostic dict: ``teacher_vocab``, ``tokenizer_vocab``,
            ``device``.
        """
        # Make the launcher-side opd_loss adapter importable inside
        # this subprocess. The Provisioner bootstrap already added
        # phase11_rlhf_grpo_infra/rlhf to sys.path; this is belt-and-
        # braces for cases where the spawn used a clean interpreter.
        if opd_loss_module_dir and opd_loss_module_dir not in sys.path:
            sys.path.insert(0, opd_loss_module_dir)

        from teacher_scorer import TeacherScorer  # noqa: E402
        from opd_loss import opd_loss  # noqa: E402
        from transformers import AutoTokenizer  # noqa: E402

        if teacher_max_memory is not None:
            logger.info(
                f"OPD init: loading teacher {teacher_model_id} "
                f"device_map=auto max_memory={teacher_max_memory}"
            )
            scorer = TeacherScorer(
                model_id=teacher_model_id,
                dtype=torch.bfloat16,
                max_memory=teacher_max_memory,
            )
        else:
            logger.info(
                f"OPD init: loading teacher {teacher_model_id} on {teacher_device}"
            )
            scorer = TeacherScorer(
                model_id=teacher_model_id,
                device=teacher_device,
                dtype=torch.bfloat16,
            )
        self._teacher_score_fn = scorer.score

        logger.info(f"OPD init: loading tokenizer from {tokenizer_path}")
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        self._opd_loss_fn = opd_loss

        v_teacher = scorer.model.lm_head.out_features
        v_tok = len(self._tokenizer)
        logger.info(
            f"OPD init done: teacher_vocab={v_teacher}, "
            f"tokenizer_vocab={v_tok}, device={teacher_device}"
        )
        return {
            "teacher_vocab": v_teacher,
            "tokenizer_vocab": v_tok,
            "device": teacher_device,
        }
