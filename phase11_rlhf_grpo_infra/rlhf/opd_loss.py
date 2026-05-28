"""GKD-loss adapter for our OPD pipeline.

REUSES the standard generalized JSD loss from TRL
(``trl.experimental.gkd.GKDTrainer.generalized_jsd_loss`` — the
Agarwal et al. 2024 GKD paper formula). We don't re-derive the loss;
we only handle (a) the student-vs-teacher vocab alignment slice
(student head padded to 163840, teacher 128256/128257; shared = 128256
Llama-3 base) and (b) label-masking so loss is computed only on
response tokens (not prompt / padding).
"""
from __future__ import annotations

import torch
from trl.experimental.gkd import GKDTrainer

# Shared vocab subset: Llama-3 base 128000 + 256 added = 128256.
# Student head is padded to 163840 (Kimi-arch default); teacher is 128256 (or
# 128257 with one extra <image> special — we still align on the safe 128256).
SHARED_VOCAB = 128256


def opd_loss(
    student_logits: torch.Tensor,   # [B, T, V_student]
    teacher_logits: torch.Tensor,   # [B, T, V_teacher]
    labels: torch.Tensor,           # [B, T] long; use -100 for positions to IGNORE
    *,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generalized JSD distillation loss on response tokens.

    Args:
        student_logits: aligned-positions student logits (response slice).
        teacher_logits: aligned-positions teacher logits (response slice).
        labels: response token ids; -100 to mask (prompt positions, padding).
        beta: 0=KL(student||teacher), 1=KL(teacher||student); 0.5=symmetric JSD.
        temperature: softmax temperature (1.0 = no scaling).

    Returns:
        Scalar loss (TRL's "batchmean" reduction).
    """
    s = student_logits[..., :SHARED_VOCAB]
    t = teacher_logits[..., :SHARED_VOCAB]
    return GKDTrainer.generalized_jsd_loss(
        student_logits=s,
        teacher_logits=t,
        labels=labels,
        beta=beta,
        temperature=temperature,
        reduction="batchmean",
    )


def _smoke() -> None:
    """Unit test: finite loss + grad masking correct."""
    torch.manual_seed(0)
    B, T = 2, 8
    student = torch.randn(B, T, 163840, requires_grad=True)
    teacher = torch.randn(B, T, 128256)
    labels = torch.randint(0, SHARED_VOCAB, (B, T))
    labels[:, :3] = -100  # first 3 positions = prompt, ignored

    loss = opd_loss(student, teacher, labels)
    assert torch.isfinite(loss).item(), "loss not finite"
    print(f"[opd] loss = {loss.item():.4f}  finite=OK")

    loss.backward()
    g = student.grad
    # 1) gradient flows on response positions in shared vocab
    g_resp_shared = g[:, 3:, :SHARED_VOCAB].abs().sum().item()
    # 2) NO gradient on masked (prompt) positions
    g_prompt = g[:, :3, :].abs().sum().item()
    # 3) NO gradient on student padding dims (sliced off)
    g_pad = g[:, :, SHARED_VOCAB:].abs().sum().item()
    print(f"[opd] grad@response shared    = {g_resp_shared:.4f}   (should be > 0)")
    print(f"[opd] grad@prompt (masked)    = {g_prompt:.6f}   (should be 0)")
    print(f"[opd] grad@student padding    = {g_pad:.6f}   (should be 0)")
    assert g_resp_shared > 0, "no grad on response positions"
    assert g_prompt == 0.0, "grad leaked into masked prompt positions"
    assert g_pad == 0.0, "grad leaked into student padding dims"
    print("[opd] SMOKE PASSED — TRL generalized_jsd_loss reusable in our pipeline")


if __name__ == "__main__":
    _smoke()
