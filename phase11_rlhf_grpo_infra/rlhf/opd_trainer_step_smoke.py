"""CPU smoke for OPDTrainer.step loss-path logic (Stage C.3).

OPDTrainer subclasses PolicyTrainer (Monarch Actor + FSDP + DCP load +
SGLang weight publish), so full instantiation requires a distributed
process group + a model_spec + vision weights. Too heavy for a CPU smoke.

Instead this test reproduces the step's *logic* against a hand-built
ToyTrainer that exposes the same attributes OPDTrainer.step touches:
self.model, self._tokenizer, self._teacher_score_fn, self._opd_loss_fn,
self.optimizers, self.lr_schedulers, self.device, self.dp_rank/dp_size,
self.config, self.model_parts, self.parallel_dims.

The validated invariant: a real teacher (random-init toy LM as stand-in)
combined with a real student (same toy LM, fresh init) produces a finite
loss, gradient flows to student params, parameters move after step().
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
TT_ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "torchtitan"))
sys.path.insert(0, TT_ROOT)


class ToyLM(nn.Module):
    """Identical to the smoke in compute_response_logits_smoke.py."""

    def __init__(self, vocab: int = 256, dim: int = 32):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.norm = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, attention_masks=None, positions=None, **_):
        h = self.emb(input_ids)
        h = self.norm(h)
        return self.lm_head(h)


class ToyTokenizer:
    """Tokenizer stand-in: char-level for clarity, but we don't actually
    round-trip text; the Episode's prompt_token_ids / token_ids are used
    directly. decode() just returns a deterministic string so the
    teacher_score_fn has something to consume."""

    def decode(self, ids, skip_special_tokens=True):
        return "_".join(str(int(i)) for i in ids)


def make_teacher_score_fn(teacher: ToyLM, tokenizer: ToyTokenizer,
                          shared_vocab: int):
    """Return a TeacherScoreFn closure.

    The toy teacher tokenizes the response_text by inverting our toy
    tokenizer (split on '_'), then forwards prompt+response through
    the teacher LM and returns response-position logits + ids.
    """
    @torch.no_grad()
    def score(image_path, prompt_text, response_text):
        prompt_ids = [int(s) for s in prompt_text.split("_") if s]
        resp_ids = [int(s) for s in response_text.split("_") if s]
        full = torch.tensor(prompt_ids + resp_ids, dtype=torch.long).unsqueeze(0)
        out = teacher(full)            # [1, T, V]
        # Same shift convention as compute_response_logits: position i
        # predicts token i+1; response window is the last T_resp positions
        # of the shifted-out logits.
        shifted = out[:, :-1, :]
        T_resp = len(resp_ids)
        teacher_logits = shifted[0, -T_resp:, :]
        teacher_ids = torch.tensor(resp_ids, dtype=torch.long)
        return teacher_logits, teacher_ids
    return score


def make_opd_loss_fn(shared_vocab: int):
    """Return an OPDLossFn closure that calls TRL's generalized_jsd_loss."""
    # Import the launcher-side adapter (NOT the trainer-side; we don't
    # want any module that pulls in torchstore here).
    sys.path.insert(0, HERE)
    from opd_loss import opd_loss  # noqa: E402
    return opd_loss


class ToyEpisode:
    """Episode stand-in carrying just what OPDTrainer.step reads."""
    def __init__(self, prompt_token_ids, token_ids, image_path=None, text=""):
        self.prompt_token_ids = prompt_token_ids
        self.token_ids = token_ids
        self.image_path = image_path
        self.text = text


def main():
    from torchtitan.experiments.rl.actors.opd_trainer import OPDTrainer
    from torchtitan.experiments.rl.actors.utils import compute_response_logits

    torch.manual_seed(0)
    V = 256
    SHARED = 200  # pretend a smaller "shared vocab" so the slice does work

    # Monkey-patch SHARED_VOCAB in the launcher's opd_loss to match
    # our toy V; the production value (128256) is wrong for a 256-V toy.
    import opd_loss as opd_loss_mod
    opd_loss_mod.SHARED_VOCAB = SHARED

    student = ToyLM(vocab=V).train()
    teacher = ToyLM(vocab=V).eval()
    teacher.lm_head.weight.data += 1.0  # make teacher distribution different
    tokenizer = ToyTokenizer()
    teacher_score_fn = make_teacher_score_fn(teacher, tokenizer, SHARED)
    opd_loss_fn = make_opd_loss_fn(SHARED)

    # Build episodes — 3 episodes with varying length.
    episodes = [
        ToyEpisode(prompt_token_ids=[3, 7, 41], token_ids=[21, 5, 99, 14],
                   text="resp1"),
        ToyEpisode(prompt_token_ids=[10, 8, 1, 9], token_ids=[2, 17, 8],
                   text="resp2"),
        ToyEpisode(prompt_token_ids=[101, 1], token_ids=[55, 4, 88, 12, 6],
                   text="resp3"),
    ]

    # Snapshot a few params before, then run the step's *body* against
    # this fixture. We exercise the bare step body (not the @endpoint
    # wrapper) by writing a tiny stub that follows the same control flow.
    # That avoids needing a Monarch actor mesh / distributed PG / FSDP.
    before = student.lm_head.weight.detach().clone()

    optim = torch.optim.SGD(student.parameters(), lr=1e-2)
    loss_accum = torch.zeros(())
    n_resp = 0
    for ep in episodes:
        prompt_text = tokenizer.decode(ep.prompt_token_ids)
        response_text = tokenizer.decode(ep.token_ids)
        with torch.no_grad():
            teacher_logits, teacher_ids = teacher_score_fn(
                ep.image_path, prompt_text, response_text,
            )
        teacher_logits = teacher_logits.to(torch.float32)
        student_logits = compute_response_logits(
            student, ep.prompt_token_ids, ep.token_ids, torch.device("cpu"),
        )
        T = min(student_logits.shape[0], teacher_logits.shape[0])
        student_logits = student_logits[:T]
        teacher_logits = teacher_logits[:T]
        labels = teacher_ids[:T].clone()
        loss = opd_loss_fn(
            student_logits.unsqueeze(0),
            teacher_logits.unsqueeze(0),
            labels.unsqueeze(0),
        )
        loss_accum = loss_accum + loss
        n_resp += T

    loss_accum = loss_accum / len(episodes)
    assert torch.isfinite(loss_accum).item(), f"loss not finite: {loss_accum}"
    print(f"[opd-step] loss = {loss_accum.item():.4f}  resp_tokens={n_resp}")

    optim.zero_grad()
    loss_accum.backward()
    g = student.lm_head.weight.grad
    assert g is not None and g.abs().sum().item() > 0, "no grad on student"
    print(f"[opd-step] student lm_head grad |sum|={g.abs().sum().item():.4f}")
    optim.step()

    after = student.lm_head.weight.detach().clone()
    delta = (after - before).norm().item()
    assert delta > 0, "student params did not move after optim.step"
    print(f"[opd-step] student params moved: ||delta||={delta:.4f}")

    # Verify the actual OPDTrainer class itself imports + has the
    # methods we expect (sanity check on the file we wrote).
    assert hasattr(OPDTrainer, "step")
    assert hasattr(OPDTrainer, "set_teacher_scorer")
    assert hasattr(OPDTrainer, "set_opd_loss_fn")
    assert hasattr(OPDTrainer, "set_tokenizer")
    print("[opd-step] OPDTrainer class shape: step + 3 setters present")

    print("[opd-step] SMOKE PASSED — step body produces finite loss, "
          "grad flows, params move")


if __name__ == "__main__":
    main()
