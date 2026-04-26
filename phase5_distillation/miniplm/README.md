# MiniPLM-style Pre-training Distillation

Implements the MiniPLM (ICLR 2025) approach for distilling Llama-3.1-8B
into our 436M Kimi Linear student via **data-side distillation**:

* No teacher forward in the training loop (vs the previous online KD
  approach which was the bottleneck and the failure mode).
* Teacher runs once, offline, on a c4-en sample to compute a per-chunk
  "difference score" relative to a small reference model. High-score
  chunks are samples where the teacher knows much more than the
  reference — that's where distillation signal is concentrated.
* Sample / weight the c4 corpus by these scores, then continue
  pretraining the student on the filtered corpus with **pure CE loss**.
  Same per-step cost as standard pretraining (no second model).

This module lives outside the torchtitan submodule on purpose — pure
workspace code, no torchtitan changes.

## Reference

Gu, Y. et al. *MiniPLM: Knowledge Distillation for Pre-Training Language
Models.* ICLR 2025. https://github.com/thu-coai/MiniPLM

## Files

(WIP, populated below as we build)

- `score_corpus.py` — offline pass: load teacher + reference, score
  c4-en chunks, write per-chunk scores to disk.
- `filter_corpus.py` — read scored chunks, sample a filtered .jsonl
  corpus by score (top-k or weighted).
- `train_continued_pretrain.sh` — torchrun launcher: continue
  pretraining the Phase 4 step-12500 student on the filtered corpus
  using the existing torchtitan kimi_linear ModelSpec + standard CE.

## Why Plan A vs the prior online KD

| | Online KD (failed) | MiniPLM (this) |
|---|---|---|
| Teacher in train loop | yes (forward each step) | no (offline once) |
| Loss | α·CE + (1-α)·T²·KL | pure CE |
| Speed | 6.3 s/step | ~3000 tps/rank (same as pretraining) |
| Result on c4 val | 3.81 (worse than pre-KD 3.73) | targeting < 2.5 |

The online KD recipe inadvertently weighted KL ~9× CE and forward-KL
made the student over-spread on the teacher's low-probability regions,
so CE didn't actually drop. MiniPLM sidesteps both problems.
