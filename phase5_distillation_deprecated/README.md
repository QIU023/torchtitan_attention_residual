# Phase 5 (deprecated) — Knowledge Distillation experiments (negative result)

> **DEPRECATED.** Renamed from `phase5_distillation/` →
> `phase5_distillation_deprecated/` after the project pivoted away
> from "use the 436M Kimi student as the multimodal LM backbone."
> The actual Phase 5 (current, multimodal AttnRes work — Path D
> Q-Former + Path F speculative draft) lives at `phase5_vlm_multimodal_sft/`. See
> `docs/multimodal_with_attn_res_design.md`.

This phase tested two paths to lower the Phase 4 Kimi Linear 436M
student's c4 val_loss past the 12,500-step pretraining floor of
**3.7326**, with the goal of producing a "multimodal-ready" backbone.

**Both attempts FAILED to improve val_loss.** The student is at the
floor for the 4× RTX 5090 + c4 + Llama-tokenizer training budget.
Multimodal work moves on without it as the LM backbone — see
`docs/multimodal_with_attn_res_design.md` for the architectural
pivot (Path D: `AttnRes Q-Former` from scratch + Path F: deploy
the AttnRes-Kimi-436M ckpt as a speculative-decoding draft model).

This dir is preserved for the negative result. Future contributors:
**don't repeat these experiments expecting a different outcome on
this hardware + data budget.**

## Results table

| approach | tokens consumed | c4 val_loss | Δ vs pre-KD floor |
|---|---|---|---|
| **pre-KD baseline** (Phase 4 step-12500) | 0.31 B | **3.7326** | 0.000 (floor) |
| Online KD (α=0.3, T=2, fwd KL with Llama-3.1-8B) | +0.16 B (10K steps) | 3.8095 | **+0.077** worse |
| MiniPLM-style data distillation (top-50% scored) | +0.44 B (18K steps) | 3.8243 | **+0.092** worse |

## Why each approach failed

### Online KD (10K steps, ~17h)

Loss formula `0.3·CE + 0.7·T²·KL` made the KL term effectively 9×
the CE weight (T²=4 multiplier). For forward-KL on a 128 K-vocab
generative model this drove the student to over-spread mass across
the teacher's low-probability regions — the canonical failure mode
that MiniLLM (NeurIPS 2023) and DistiLLM-2 (ICML 2025 oral) both
call out. The KL term shrank during training (intended signal) but
the CE term — which is what c4 val_loss measures — barely moved.

**SOTA recipe would have been**: reverse-KL or skew-KL, no CE term,
on-policy generation. Implementing that would have been a 2-3× more
invasive change. We did not pursue it after the simpler MiniPLM
data path turned out to be an equally interesting (and equally
failing) 2024-2025 SOTA direction.

Original teacher choice **Kimi-Linear-48B-A3B-Base** (96 GB
download, FSDP-shardable across 4 ranks) was abandoned mid-debug
when we discovered the Phase 4 student was actually trained with
**Llama-3.1 tokenizer** (vocab 128,256), not the Kimi BPE the
config flavor advertises. Switched teacher to
`NousResearch/Meta-Llama-3.1-8B` (15 GB, non-gated redistribution)
to get a vocab match.

### MiniPLM data distillation (18K steps, ~9.5h)

The data-side approach: score 120K c4-en chunks by
`log p_teacher(chunk) - log p_reference(chunk)` (teacher
Llama-3.1-8B, reference Llama-3.2-1B), keep the top 50%, run
standard CE pretraining on the filtered subset. Per the MiniPLM
ICLR 2025 paper this is the recommended *pretraining-stage*
distillation paradigm — no teacher in the train loop, runs at
full pretraining throughput.

We hit **3,224 tps/rank with compile** (== Phase 4 baseline
throughput, confirming the data + pipeline were sound). Training
loss EMA dropped from 4.13 to ~3.30 over 18K steps. But c4 val_loss
came back at **3.82 — worse than the unfiltered baseline of 3.73**.

**Diagnosis**: MiniPLM's score function selects chunks where the
8B teacher knows much more than the 1B reference. For our 436M
student with effectively zero-budget for distillation, those are
exactly the chunks the small student CAN'T learn well. We overfit
to filtered hard chunks while losing coverage of easier content,
net val degradation. MiniPLM's intended recipient (in their paper)
was a 200M-1.2B student trained from scratch at larger compute —
not a 436M ckpt being fine-tuned on a tight budget.

## Bugs surfaced and fixed during development (preserved for record)

* **Off-by-1 in `LocalJsonlIterableDataset`** caused infinite NCCL
  hang at start of training — checked `len(ids) >= seq_len + 1` but
  scored chunks were exactly `seq_len`. Fixed via stream
  re-windowing across chunk boundaries (torchtitan-standard
  pattern). Took ~3 false starts to localize.
* **DataLoader `num_workers > 0` after Trainer.__init__** corrupted
  CUDA context via fork — same fork-after-init NCCL issue
  torchtitan's main path avoids by defaulting to `num_workers=0`.
* **Tee buffering hides early step logs in train.log** — actual
  output reaches torchelastic per-rank stdout (`/tmp/torchelastic_*`)
  in real time. Run dirs now ship a `rank0_stdout.log` symlink.
* **ConfigManager.parse_args default-arg binding** captures
  `sys.argv` at function-definition time, not call time. Custom
  pre-parsers must pass `sys.argv[1:]` explicitly.
* **`torchtitan` logger needs `init_logger()`** explicitly when
  bypassing `torchtitan/train.py` main entry — otherwise INFO logs
  silently drop, only WARN+ visible.

## Files

* `kd_loss.py` — KD loss math (`α·CE + (1-α)·T²·KL`). 10 unit tests.
  Standalone module, no torchtitan dependency.
* `tests/test_kd_loss.py` — KD loss tests.
* `teacher_runner.py` — HF `AutoModelForCausalLM` + FSDP2 wrapper.
* `train_kd.py` — online KD training script (failed first attempt).
* `eval_kd_student.sh` — c4_validation eval after KD training.
* `launch_kd.sh` — torchrun launcher for `train_kd.py`.
* `runs/kd_student_eval/eval.log` — final online-KD eval log
  (val_loss 3.8095).
* `miniplm/` — full MiniPLM-style data distillation pipeline
  (score → filter → continue pretrain → eval).
  See `miniplm/README.md` for design notes; ckpts in `runs/.../checkpoint/`
  gitignored.

## Recommendations for future work in this dir

If you return to LLM distillation with this same student:

1. **Don't re-run our recipes.** Both have been measured at this
   hardware + data budget. Same setup → same negative result.
2. **Try MiniLLM** (reverse KL + on-policy) if you can afford the
   3-4× engineering cost. MiniPLM and online forward-KL are the
   wrong primitive for small students.
3. **Try non-c4 data** (instruction tuning, code, math) before
   spending more compute on c4 filtering — distribution shift
   matters more than data ranking at this scale.
4. **Consider giving up on the 436M as backbone** for downstream
   tasks. Use a larger frozen public LLM (Llama-3.1-8B-Base or
   Qwen3-7B-Base), apply AttnRes only to from-scratch trained
   components (vision↔LLM connector). See
   `docs/multimodal_with_attn_res_design.md` for that pivot.
