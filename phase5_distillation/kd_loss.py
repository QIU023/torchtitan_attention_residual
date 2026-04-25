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
        student_logits: ``(B, T, V)`` student forward output, fp32-safe
            but typically bf16; cast to fp32 inside for numerical
            stability.
        labels: ``(B, T)`` integer gold tokens. Positions equal to
            ``cfg.ignore_index`` are skipped in both CE and KL.
        teacher_logits: ``(B, T, V)`` pre-computed teacher output.
            Must be ``torch.no_grad()`` upstream so no teacher
            gradient is built. Shape and vocab must match the student.
        cfg: KDConfig controlling ``alpha``, ``temperature``,
            ``ignore_index``. ``None`` uses defaults.

    Returns:
        Scalar loss, sum-reduced over non-ignored tokens (matching the
        convention of ``components.loss.cross_entropy_loss``). Divide
        by ``(labels != ignore_index).sum()`` upstream to get the
        per-token mean.
    """
    if cfg is None:
        cfg = KDConfig()

    assert student_logits.shape == teacher_logits.shape, (
        f"student/teacher logits shape mismatch: "
        f"{tuple(student_logits.shape)} vs {tuple(teacher_logits.shape)}"
    )
    assert labels.shape == student_logits.shape[:-1], (
        f"labels shape {tuple(labels.shape)} incompatible with "
        f"student logits {tuple(student_logits.shape)}"
    )

    # Flatten (B, T) -> (B*T,) for masking simplicity.
    logits_s = student_logits.flatten(0, 1).float()  # (N, V)
    logits_t = teacher_logits.flatten(0, 1).float()  # (N, V)
    targets = labels.flatten(0, 1)                    # (N,)

    # CE term — sum-reduction, ignores IGNORE_INDEX positions.
    ce = F.cross_entropy(
        logits_s, targets,
        reduction="sum",
        ignore_index=cfg.ignore_index,
    )

    # KL term — keep only non-ignored positions.
    keep = targets.ne(cfg.ignore_index)
    if not keep.any():
        return cfg.alpha * ce  # all masked; KL contributes 0

    T = cfg.temperature
    log_p_s = F.log_softmax(logits_s[keep] / T, dim=-1)
    log_p_t = F.log_softmax(logits_t[keep] / T, dim=-1)
    # KL(student || teacher) = sum p_s * (log p_s - log p_t).
    # Equivalently: F.kl_div(log_p_t, log_p_s, log_target=True,
    # reduction='sum') — note F.kl_div expects (input=log_p_target,
    # target=log_p_source) when log_target=True. We use the explicit
    # form to keep the direction obvious.
    p_s = log_p_s.exp()
    kl = (p_s * (log_p_s - log_p_t)).sum()
    # Hinton rescaling — T^2 keeps KL grad magnitude commensurate with CE.
    kl = kl * (T * T)

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
