# Phase 5 — Distillation from Kimi-Linear-48B-A3B-Base into the 436M student

Standalone module that lives **outside** the torchtitan submodule. Distillation
is a project-level concern (teacher loading, two-model orchestration, separate
training loop) rather than a torchtitan-core feature; keeping it out of
`torchtitan/torchtitan/experiments/` keeps the submodule's role clear: pure
training framework for the student.

## Why distillation here

Continued CE pretraining of the 436M student to a multimodal-ready loss
(~2.5–3.0) needs ~10B+ tokens, ~100 GPU-days on 4× RTX 5090. Distilling from
**Kimi-Linear-48B-A3B-Base** as teacher reaches the same loss target with
~1–3B KD tokens (1–3 days on the same hardware) because the dense softmax
target carries 5–10× more signal per token than CE on a 1-hot label.

See `docs/pretraining_closure_and_kd_plan.md` for the broader plan.

## Why this teacher

| candidate | tokenizer | shares vocab with student? |
|---|---|---|
| Qwen3 / Qwen3.5 | Qwen BPE (vocab 152,064) | ❌ |
| Llama3 | Llama (128,256) | ❌ |
| DeepSeek-V3 | DSv3 BPE (129,280) | ❌ |
| **Kimi-Linear-48B-A3B-Base** | **Kimi BPE (163,840)** | ✅ |

KD on token-level logits requires student and teacher to share **the same
vocab**, otherwise `KL(softmax_s ‖ softmax_t)` is not well-defined. Kimi
tokenizer is unique to the Kimi family, so the teacher has to be in that
family.

## Memory plan on 4× RTX 5090

48B params bf16 = 96 GB total. With FSDP2 sharding across 4 ranks:
~24 GB / rank for weights. Forward-only inference on the teacher means no
optimizer state, no gradient accumulators. Activations are small (B=1
T=2048 D=2304 → ~4 MB / layer). int8 weight-only quantization halves to
12 GB / rank, giving comfortable headroom for the student's own footprint.

## Layout

- `kd_loss.py` — `kd_loss(student, labels, teacher, cfg)` math module.
  Pure torch, no torchtitan dependency. Implements
  `α·CE + (1-α)·T²·KL(student ‖ teacher)` with Hinton T² rescaling.
- `tests/test_kd_loss.py` — 9 unit tests covering α=0, α=1, identical
  teacher, ignore-index masking, temperature, backward path.

## Pending (not yet written)

- `teacher_runner.py` — wraps a HF `AutoModelForCausalLM` in eval / no_grad
  mode, FSDP-sharded across the 4 GPUs, exposes `.forward(input_ids) →
  logits`. Uses `transformers` directly (NOT vLLM — vLLM only exposes top-K
  logprobs, full-vocab softmax is required for KD's KL term).
- `train_kd.py` — driver that:
  1. Builds the student via torchtitan's `kimi_linear` ModelSpec, loads the
     12,500-step (or 30K/60K-closure) ckpt.
  2. Builds the teacher via `teacher_runner`.
  3. Runs the same input batch through both, computes `kd_loss`, optimizes
     student only.
- `launch_kd.sh` — torchrun wrapper, mirrors `phase4/launch_fsdp_small.sh`
  but adds teacher loading.
