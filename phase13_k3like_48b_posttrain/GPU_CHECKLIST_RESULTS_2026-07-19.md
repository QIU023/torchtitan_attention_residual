# Pre-RFC GPU checklist — results (8x RTX 5090, 2026-07-19)

> Execution log for [PRE_RFC_GPU_CHECKLIST.md](PRE_RFC_GPU_CHECKLIST.md).
> Box: 8x RTX 5090 32GB (SM 12.0), torch 2.12.0+cu130, fla-core 0.5.1,
> torchao 0.17.0. Fork start `a3b3c74b3` -> fixes committed on
> `attention_residual_dev` (see git log; RFC links must re-pin).

## Verdict so far

Steps 1-3 GREEN after fixes. Step 4 L16 matrix GREEN (one shape needed
the fp32 exoneration below). kimi48b pair: naive OOMs at seq1024 -- the
seq512 fair pair is running; adapter-only seq1024 completes (memory
evidence in itself). Step 5 pending.

## Step 1 — full suites

- kimi_k3: **44 passed, 0 skipped** (was 43+2 hardware skips on CPU).
- dense_carrier: **70 passed, 0 skipped** (was 64+6).
- Fixes required:
  - fp8 flavor: `Float8LinearConverter.convert` (config-tree traversal)
    can never apply to the plain-module KimiLinear model; replaced with
    module-level torchao swap in `KimiLinearFloat8Spec.build()`. The old
    `filter_fqns` ("kda", "mla.q_lora_proj") matched NOTHING in the real
    module tree -- KDA now excluded structurally, MLA low-rank via its
    real name (`kv_a_proj_with_mqa`). fp8 build verified: 60 linears
    swapped, KDA/kv_a/AttnRes/heads kept bf16.
  - DSv3 CUDA tests had never run on a CUDA box (model built on CPU
    under a CUDA-only gate); moved build+tokens to CUDA.

## Step 2 — 5-step train smokes

Both complete (kimi 194m loss 12.18, fla triton real path; dense 175m
loss 12.20). Four merged-Trainer-chain breaks fixed:
`update_from_config` kwarg rename, `traverse` requirement on spec shims
(+ honest fp8 marker for `has_quantization`/MFU suppression), concrete
CE loss wiring in both registries (plain CE = historical numerics;
ChunkedLossWrapper needs `_skip_lm_head` the Kimi model lacks),
`_set_pg_timeout` for torch 2.12 stable.

## Step 3 — 4-GPU PP smokes (100 steps)

naive 8.1255 / adapter 8.1066, both complete. Fixes:
`_generate_llm_fqn_per_model_part` privatization; per-step loss kwargs
holder (torch 2.12 `step()` has no `loss_kwargs` -- dropping them would
silently unscale PP gradients); **shape-inference delta placeholder must
mirror runtime `requires_grad`** -- torch 2.12 derives recv-buffer and
grad-send metadata from shape inference, so `new_zeros()` (rg=False)
severed every cross-stage delta backward edge
(PipeliningMetadataError at SEND_B; found via per-stage grad probe:
last stage `in[1] rg=False`).

## Step 4 — pressure matrix

L16_n8, 1000 steps, 8 GPUs, bf16 (`pressure_test_20260719-0801`):

| shape | naive | adapter | final |dLoss| | late signed mean | verdict |
|---|---|---|---|---|---|
| PP8xVP2 (LPS=1) | 5.27563 | 5.26996 | 0.0057 | +0.0016 (31/51) | PASS |
| PP4xVP2 (LPS=2, DP=2) | 5.47948 | 5.44676 | 0.0327 | +0.0299 (51/51) | PASS via fp32 probe |
| PP4xVP4 (LPS=1, DP=2) | 5.11195 | 5.10225 | 0.0097 | +0.0076 (45/51) | PASS |

Step-1 losses identical across all six runs (11.7618): init + data
order seed-matched; AttnRes zero-init identity holds.

**PP4xVP2 fp32 exoneration.** The 0.0327 exceeds the historical 0.011
band AND is same-signed at every late step -- not dismissible as noise
by inspection. Discriminating probe: same shape, 100 steps,
`--training.mixed_precision_param float32 --debug.seed 42
--debug.deterministic`, naive vs adapter:

- loss IDENTICAL to 5 decimals through step ~90; max |dLoss| = 3e-5
  (step 96), mean 1e-6.

The adapter graph is numerically equivalent; the bf16 gap is
reordering-noise trajectory divergence (LPS=2 changes the adapter's
block-stack assembly order). A bf16 seed-vs-seed control (naive seed
123 vs baseline naive) is running to calibrate the band on this
box/torch; result to be appended.

**kimi48b d1280 e16 L32N8 PP8xVP4 (300 steps, LBS=32):**

- seq1024 naive: **OOM** (GPU1, 29.9 GiB allocated; needs ~40 MiB more).
  Repo history contains only adapter logs at this shape -- consistent:
  the naive full-stack transfer does not fit 32 GB cards. The adapter's
  memory saving is what makes this shape feasible at all.
- seq1024 adapter: completes, loss 5.97096 @300, peak 27.15 GiB (86.6%).
- seq512 naive-vs-adapter fair pair: running; result to be appended.

## Step 5 — pending

447m TP=2 / EP=2 smokes + fp8 flavor 5-step smoke queued after step 4
runs drain.

## Known-risk notes

- fla `fused_norm_gate` device-side assert (SM 12.0, ~650-step KDA
  threshold): NOT observed in any run so far (longest KDA run: 300
  steps; L16 dense runs don't exercise KDA).
- torch 2.12-stable vs upstream-nightly drift is the dominant breakage
  class this pass (loss_kwargs, set_timeout, shape-inference metadata).
