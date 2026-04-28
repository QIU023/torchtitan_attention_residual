# Phase 2 Report — Single-GPU Block AttnRes Loss-Curve Alignment

**Date**: 2026-04-19 (main runs) / 2026-04-20 (ablation sweep) / report 2026-04-28
**Status**: **DONE** — primary loss-curve evidence collected; both A/B and N-sweep ablation completed.
**Hardware**: 1× RTX 5090 (32 GB), single GPU.

---

## 1. Goal

Train a 175 M Llama3 baseline and a 175 M Block AttnRes variant on identical hyperparameters, then show that the AttnRes train-loss curve sits visibly below baseline. This is the correctness gate that has to land before the Phase-3 PP work or the RFC PR can claim anything.

In addition, sweep `num_blocks` ∈ {3, 6, 12} so the RFC PR description has a small ablation showing N≈8 (the paper's headline) is not load-bearing for "AttnRes < baseline" at this scale.

---

## 2. What shipped

**Workspace scripts** (`phase2/`, NOT part of the torchtitan PR):

| File | Role |
|---|---|
| `setup_env.sh` | conda env `attnres`, torch nightly, torchtitan editable, Llama-3.1 tokenizer pull (ungated `NousResearch/Meta-Llama-3.1-8B` mirror by default), runs the standalone smoke + torchtitan unit tests |
| `smoke_test_attn_res.py` | torch-only inline reimpl of `block_attn_res()`. 4 self-checks: identity (single-partial), uniform softmax with zero-init proj, gradient flow, multi-layer block commit, finite output / loss / gradients with zero-init pseudo-queries |
| `launch.sh` | tmux 4-window launcher: `baseline` → `guardian` (waits for `Training completed` in baseline log, touches `DONE`) → `attn_res` (gated on `DONE`) → `monitor` (`nvidia-smi`). `DATA_FRAC` env scales steps for shorter dry runs (used for `runs_1_8th/`) |
| `launch_ablation.sh` | unconditional chain (`;` not `&&`) so a mid-run crash of variant N doesn't block N+1; per-variant `STATUS` file with the torchrun rc; default variants: `n3` + `n12` (two ends of the sweep, primary `n6` already in `runs/attn_res/`) |
| `compare_losses.py` | reads TB events from baseline + attn_res dirs, produces 3-panel figure (full curves / post-warmup zoom / per-step delta), prints same-step delta milestones |
| `plot_ablation.py` | overlay plot for the N-sweep |
| `README.md` | runbook (env, dry-run validation, smoke, main run, monitor, compare, troubleshooting) |

**Committable code** (`torchtitan/experiments/attn_res/`):

| File | Role |
|---|---|
| `attn_res.py` | `block_attn_res()` primitive (paper Figure 2), `AttnResProjection` (D→1, zero-init pseudo-query), `stack_blocks` / `unstack_blocks` |
| `model.py` | `AttnResLlama3TransformerBlock` and `AttnResLlama3Model` subclasses — core `decoder.py` / `model.py` untouched |
| `__init__.py` | model flavors: `debugmodel_attn_res`, `175M_attn_res` (default `N=6`), `175M_attn_res_n{3,4,12}`, `175M_attn_res_L16_n8` |
| `config_registry.py` | trainer configs: `llama3_175m_baseline`, `llama3_175m_attn_res` |
| `tests/test_attn_res.py` | unit tests (primitive, projection, stack/unstack, dense model, decoder integration, init contract) |

---

## 3. Setup (committed in scripts)

| Knob | Value | Source |
|---|---|---|
| Model | Llama3-175M (custom flavor) | `torchtitan/experiments/attn_res/__init__.py` |
| Tokenizer | Llama-3.1 (NousResearch ungated mirror) | `setup_env.sh` |
| Dataset | C4 (HF stream) | torchtitan default |
| `local_batch_size` | 8 | `launch.sh` (config default 16 OOMs xent on 5090 since logits are `[B*T, V=128256]` fp32 — gradient accumulation keeps effective bs intact) |
| `global_batch_size` | 16 | `launch.sh` (grad accum = 2) |
| Sequence length | 2048 | torchtitan default |
| Steps | 20 000 | `launch.sh` default |
| Seed | torchtitan default | not pinned across baseline/AttnRes (deliberately, as a noise floor sample) |
| `dtype` | bf16 | torchtitan default |
| Optimizer | AdamW | torchtitan default |
| LR schedule | cosine, 200-step warmup, default decay | torchtitan default |

---

## 4. Validated runs

### 4.1 Primary A/B (`runs/baseline` vs `runs/attn_res`)

20 000 steps each, ≈ 650 M tokens per run.

| step | baseline loss | attn_res loss | delta |
|---:|---:|---:|---:|
| 1 | 12.26907 | 12.26127 | −0.008 |
| 990 | 5.26878 | 5.17480 | −0.094 |
| 4990 | 4.24073 | 4.16045 | −0.080 |
| 9990 | 4.09605 | 4.03300 | −0.063 |
| 19990 | 3.88939 | 3.84742 | −0.042 |
| 20000 | 3.68482 | 3.61859 | **−0.066** |

Both runs printed `Training completed`. Delta is consistently negative through training, plot saved at `runs/comparison.png`.

Wall clock and throughput:
- Baseline: ~2h42m on 5090, ~71.2 K tps, MFU ~15.5 %, peak mem 29.1 GiB.
- AttnRes: ~3h41m, ~50.1 K tps, MFU ~10.9 %, peak mem 30.05 GiB.

The ~30 % tps cost is expected (paper §3.2): every sub-layer now does a stack + RMSNorm + einsum + softmax + weighted sum over (N+1) block activations. The gap should largely collapse under PP (cross-stage caching) and `torch.compile` (out of scope here).

### 4.2 Sanity-scale dry run (`runs_1_8th/`)

`DATA_FRAC=0.125` → 2 500 steps each. Final losses:
- baseline 4.82641
- attn_res 4.71197 (delta −0.114 at the same step)

Same direction, larger magnitude (early curve), confirms the launch path works end-to-end before kicking the overnight run.

### 4.3 N-sweep ablation (`runs/ablation/`)

20 000 steps each, identical hyperparams except `num_blocks`.

| variant | num_blocks | layers_per_block | step 20000 loss | tps | MFU | mem |
|---|---:|---:|---:|---:|---:|---:|
| `attn_res_n3` | 3 | 8 | **3.65491** | 52 664 | 11.5 % | 29.88 GiB |
| `attn_res` (n6) | 6 | 4 | **3.61859** | 49 412 | 10.8 % | 30.05 GiB |
| `attn_res_n12` | 12 | 2 | **3.62343** | 26 437 | 5.8 % | 29.90 GiB |
| `baseline` | — | — | 3.68482 | 70 660 | 15.4 % | 29.11 GiB |

All three N values beat baseline at step 20 000. N=6 (paper's "≈8" sweet spot for L=24 in our setup) wins by a hair; N=3 only ~0.04 worse, N=12 sits between. Throughput collapses at N=12 (the per-sub-layer stack grows linearly in N — 12 cached blocks dominate the layer cost).

The first `n12` attempt (`llama3_175m_attn_res_n12_crashed_20260419/`) died on a transient HF httpx error (`Cannot send a request, as the client has been closed`) mid-run; the unconditional-chain ablation launcher carried on, the rerun (`llama3_175m_attn_res_n12/`) finished cleanly and replaced the crashed dir's role.

### 4.4 Standalone smoke + unit tests

`smoke_test_attn_res.py` and `torchtitan/experiments/attn_res/tests/test_attn_res.py` both pass under `setup_env.sh`. Catches:

- core primitive identity (single-partial → identity-like with zero-init proj)
- uniform softmax behaviour with zero-init pseudo-query → step-0 numerically equivalent to standard residuals
- gradient flow through stack/cat
- multi-layer block commit (3 committed blocks + 1 partial → 4-source aggregation)
- finite output / loss / gradients on every layer's params

---

## 5. Findings

1. **Block AttnRes is correctly implemented at the primitive level.** The end-of-training delta (−0.066 train loss at 650 M tokens) is the proof-of-correctness artifact for the RFC PR. Direction matches the paper's "AttnRes ≈ baseline × 1.25 compute at matched size" claim qualitatively; absolute magnitude isn't comparable since paper uses a different scaling-law setup.
2. **Zero-init pseudo-queries matter.** The unit test `test_pseudo_queries_are_zero_after_init` explicitly checks this. A non-zero pseudo-query at step 0 produces a non-uniform softmax → effectively scrambled residual at init → training volatility / NaN under bf16.
3. **N is not very load-bearing at 175 M / 650 M tokens.** N ∈ {3, 6, 12} all beat baseline; N=6 wins by ~0.04 over N=3 and ~0.005 over N=12. The headline value of "N≈8" in the paper at this scale is small; the engineering value (Block AttnRes vs Full AttnRes memory/comm) lives at PP scale (Phase 3).
4. **Per-step compute overhead at single-GPU: ~30 %.** Acceptable for a correctness experiment; will need to recover under PP for the RFC PR's #2 headline.

## 6. What this unlocks

- Phase 3 (PP cache adapter): correctness baseline established.
- RFC PR description: `comparison.png` + final-loss tail-of-log + param count are the three artifacts that go into the PR write-up.
- Block AttnRes flavors registered in `attn_res/__init__.py` are reusable for Phase 4 (Kimi Linear backbone) without modification.

## 7. Pointers

- Run logs: `phase2/runs/{baseline,attn_res}/train.log`, `runs/ablation/*/train.log`.
- Plots: `phase2/runs/comparison.png`, `runs/ablation/comparison.png`, `runs_1_8th/comparison.png`.
- Smoke test: `phase2/smoke_test_attn_res.py`.
- Runbook: `phase2/README.md`.
- Code: [torchtitan/experiments/attn_res/](../../torchtitan/torchtitan/experiments/attn_res/).
