# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Knowledge distillation loss module for the Kimi Linear student.

Lives outside the torchtitan submodule on purpose: distillation is a
project-level concern (teacher loading, KD orchestration, separate
training script), not a torchtitan-core feature. torchtitan provides
the student forward / FSDP / PP plumbing; this module composes on top.

Teacher-agnostic: accepts pre-computed teacher logits of the same
shape and vocabulary as the student, applies the standard
KD-interpolation loss:

    L = alpha * CE(student_logits, gold_tokens)
      + (1 - alpha) * T^2 * KL( softmax(student/T) || softmax(teacher/T) )

Caller responsibilities:

1. Produce teacher logits in-step under ``torch.no_grad()`` on the
   same input tokens. Pre-computed offline logits are not supported
   (storage prohibitive: 163K-vocab × 1B tokens >> 600GB).
2. Ensure tokenizer / vocab alignment between student and teacher.
   For Kimi Linear (vocab 163840) the natural teacher is
   Kimi-Linear-48B-A3B-Base.
3. Keep ``teacher_logits`` in scope through the student's backward.

See ``docs/pretraining_closure_and_kd_plan.md`` for the overall plan.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

# torchtitan's IGNORE_INDEX is just -100; replicate the constant
# locally so this module has no torchtitan dependency.
IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class KDConfig:
    """Config for token-level logit KD.

    Attributes:
        alpha: Weight on the gold-label CE term. The KD-KL term gets
            ``(1 - alpha)``. Paper recipes use ``alpha in [0.1, 0.5]``;
            default 0.3 matches the plan in
            ``docs/pretraining_closure_and_kd_plan.md``.
        temperature: Softmax temperature T applied to both student
            and teacher logits before the KL. T=1 is plain KL over
            the raw distribution; T>1 smooths peaks so the student
            learns non-argmax mass. Default 2.0. The KL term is
            rescaled by T^2 to keep gradient magnitudes comparable
            to CE (Hinton et al. 2015 §2.1).
        ignore_index: Label value to skip in CE. Teacher positions at
            the same index are also masked out of the KL.
    """

    alpha: float = 0.3
    temperature: float = 2.0
    ignore_index: int = IGNORE_INDEX


def kd_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: torch.Tensor,
    cfg: KDConfig | None = None,
) -> torch.Tensor:
    """Token-level logit KD loss.

    Args:
        student_logits: ``(B, T, V_s)`` student forward output, fp32-safe
            but typically bf16; cast to fp32 inside for numerical
            stability.
        labels: ``(B, T)`` integer gold tokens. Positions equal to
            ``cfg.ignore_index`` are skipped in both CE and KL.
        teacher_logits: ``(B, T, V_t)`` pre-computed teacher output.
            Must be ``torch.no_grad()`` upstream so no teacher
            gradient is built.
        cfg: KDConfig controlling ``alpha``, ``temperature``,
            ``ignore_index``. ``None`` uses defaults.

    Vocab handling:
        * If ``V_s == V_t`` — standard same-vocab KD.
        * If ``V_s > V_t`` — student vocab is a superset of teacher's
          (typical case: torchtitan's kimi_linear configs use
          vocab=163840 placeholder, but the student was trained with
          a smaller-vocab tokenizer so the upper rows are unused).
          Student logits are sliced to ``[:V_t]`` for both CE and KL
          so both terms share a single normalization basis.
        * If ``V_s < V_t`` — not supported (student couldn't have been
          trained on those teacher tokens).

    Returns:
        Scalar loss, sum-reduced over non-ignored tokens. Divide by
        ``(labels != ignore_index).sum()`` upstream to get per-token mean.
    """
    if cfg is None:
        cfg = KDConfig()

    V_s = student_logits.shape[-1]
    V_t = teacher_logits.shape[-1]
    if V_s < V_t:
        raise ValueError(
            f"Student vocab ({V_s}) smaller than teacher ({V_t}); "
            f"cannot align KL distribution."
        )
    if student_logits.shape[:-1] != teacher_logits.shape[:-1]:
        raise ValueError(
            f"Batch / seq mismatch: student {tuple(student_logits.shape)} "
            f"vs teacher {tuple(teacher_logits.shape)}"
        )
    if labels.shape != student_logits.shape[:-1]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} incompatible with "
            f"student logits {tuple(student_logits.shape)}"
        )

    # Flatten (B, T) -> (B*T,). For V_s > V_t, slice student to V_t so
    # CE and KL share a single softmax basis (avoids double-bookkeeping
    # of the unused upper rows that were never trained anyway).
    if V_s > V_t:
        student_logits = student_logits[..., :V_t]
    # Keep math in the input dtype (typically bf16) — fp32 promotion
    # of the [N, V_t] vocab tensor would peak ~2x VRAM and OOMs the
    # student×teacher KD step on 31 GiB cards. F.cross_entropy and
    # F.log_softmax handle bf16 with built-in stable max-subtraction,
    # so we skip the explicit .float() upcast.
    logits_s = student_logits.flatten(0, 1)  # (N, V_t)
    logits_t = teacher_logits.flatten(0, 1)  # (N, V_t)
    targets = labels.flatten(0, 1)            # (N,)

    # CE term — sum-reduction, ignores IGNORE_INDEX positions. Cast
    # only the CE path to fp32 (cross_entropy_loss in torchtitan does
    # the same); the KL path stays in bf16.
    ce = F.cross_entropy(
        logits_s.float(), targets,
        reduction="sum",
        ignore_index=cfg.ignore_index,
    )

    # KL term — keep only non-ignored positions.
    keep = targets.ne(cfg.ignore_index)
    if not keep.any():
        return cfg.alpha * ce  # all masked; KL contributes 0

    T_temp = cfg.temperature
    # Slice keep BEFORE casting — slicing avoids materializing a
    # full-size fp32 buffer for masked positions that we discard.
    log_p_s = F.log_softmax(logits_s[keep] / T_temp, dim=-1)
    log_p_t = F.log_softmax(logits_t[keep] / T_temp, dim=-1)
    # KL(student || teacher) = sum p_s * (log p_s - log p_t).
    p_s = log_p_s.exp()
    kl = (p_s * (log_p_s - log_p_t)).sum()
    # Hinton T^2 rescaling so KL grad magnitude stays commensurate
    # with CE across temperature changes.
    kl = kl * (T_temp * T_temp)

    return cfg.alpha * ce + (1.0 - cfg.alpha) * kl


def build_kd_loss(cfg: KDConfig | None = None):
    """Factory returning a closure with signature
    ``(student_logits, labels, teacher_logits) -> loss``.

    Matching the shape used by existing torchtitan loss builders
    (see ``components/loss.py``). Does NOT wrap with torch.compile —
    the caller should compose that separately if desired, as KD
    loss is called once per step vs per-microbatch.
    """
    _cfg = cfg or KDConfig()

    def _fn(
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        return kd_loss(student_logits, labels, teacher_logits, _cfg)

    return _fn
