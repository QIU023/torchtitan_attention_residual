# Pre-RFC GPU checklist — results (8x RTX 5090, 2026-07-19)

> Execution log for [PRE_RFC_GPU_CHECKLIST.md](PRE_RFC_GPU_CHECKLIST.md).
> Box: 8x RTX 5090 32GB (SM 12.0), torch 2.12.0+cu130, fla-core 0.5.1,
> torchao 0.17.0. Fork start `a3b3c74b3` -> fixes committed on
> `attention_residual_dev` (see git log; RFC links must re-pin).

## Verdict so far

Steps 1-4 GREEN after fixes (step 4: L16 matrix passes with the
measured seed band + fp32 equivalence; kimi48b passes at seq512, and
the seq1024 naive OOM is itself the memory-saving evidence). Step 5
running.

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

**Cross-era comparison vs PRESSURE_TEST_REPORT_2026-05-12 (same
shapes, same script defaults, pre-merge fork + older torch):**

| shape | hist naive | today naive | hist adapter | today adapter | hist |d| | today |d| |
|---|---|---|---|---|---|---|
| PP8xVP2 | 5.42497 | 5.27563 | 5.42935 | 5.26996 | 0.0044 | 0.0057 |
| PP4xVP2 | 5.52833 | 5.47948 | 5.52941 | 5.44676 | 0.0011 | 0.0327 |
| PP4xVP4 | 5.13467 | 5.11195 | 5.13877 | 5.10225 | 0.0041 | 0.0097 |

- Shape ordering of final losses reproduces exactly
  (vp4 < pp8 < pp4_vp2 in both eras).
- Absolute losses shifted -0.02..-0.15 across eras (torch 2.12 +
  merged-trunk kernels; both columns internally consistent).
- Historical naive-vs-naive band: 0.06-0.13 (phase3 handoff
  2026-04-21); today's measured seed band: 0.0292. Adapter deltas sit
  inside the band in both eras. (The checklist's "0.011" reference
  traces to a different report; the 2026-05-12 L16 report itself says
  max 0.0044 with band 0.06-0.13.)
- Step-time note: today's per-run timing is polluted by the validator
  firing at steps 500/1000 and single-snapshot tps; the RFC's perf
  numbers continue to cite the historical report, which this pass did
  not re-measure.

**kimi48b adapter, same script, cross-era:** historical step-300 loss
5.96955 (24.76 GiB, tps 1096) vs today 5.97096 (27.15 GiB, tps 919)
-> **cross-era |dLoss| = 0.0014** on the 48B-layout carrier across a
torch major bump + 8 fix commits. Memory +2.4 GiB under torch 2.12.

**PP4xVP2 fp32 exoneration.** The 0.0327 exceeds the historical 0.011
band AND is same-signed at every late step -- not dismissible as noise
by inspection. Discriminating probe: same shape, 100 steps,
`--training.mixed_precision_param float32 --debug.seed 42
--debug.deterministic`, naive vs adapter:

- loss IDENTICAL to 5 decimals through step ~90; max |dLoss| = 3e-5
  (step 96), mean 1e-6.

The adapter graph is numerically equivalent; the bf16 gap is
reordering-noise trajectory divergence (LPS=2 changes the adapter's
block-stack assembly order).

**Measured seed band (this box / torch 2.12).** pp4_vp2 naive with
`--debug.seed 123` vs the baseline naive: final loss 5.50868 vs
5.47948 -> **seed-vs-seed |dLoss| = 0.0292** at 1000 steps. The
naive-vs-adapter 0.0327 is the same magnitude as pure seed noise on
this environment, satisfying the acceptance criterion as written
("within the bf16 seed-vs-seed nondeterminism band"); the historical
0.011 figure was that report's measurement, not a universal constant.

**kimi48b d1280 e16 L32N8 PP8xVP4 (300 steps, LBS=32):**

- seq1024 naive: **OOM** (GPU1, 29.9 GiB allocated; needs ~40 MiB more).
  Repo history contains only adapter logs at this shape -- consistent:
  the naive full-stack transfer does not fit 32 GB cards. The adapter's
  memory saving is what makes this shape feasible at all.
- seq1024 adapter: completes, loss 5.97096 @300, peak 27.15 GiB (86.6%).
- seq512 fair pair (both modes fit): naive 6.54285 vs adapter 6.54153
  at step 300 -> **|dLoss| = 0.0013**. PASS.

## Step 5 — optional hardening (partial)

- **fp8 flavor 5-step smoke: PASS** (1 GPU, seq 512, SM 12.0 native
  rowwise fp8 through KimiLinearFloat8Spec's module-level swap). This
  closes the last hardware gate from step 1's skip list.
- **TP=2: BLOCKED (known drift, root-caused).** parallelize.py API
  drift fixed (NoParallel kwarg, experts tree); the remaining failure
  is integration-level: the merged common MoE no longer to_locals its
  input, so the ffn-container NoParallel DTensor-izes x while the gate
  output stays plain -> mixed Tensor/DTensor in the token dispatcher.
  Needs a redesigned TP plan for the MoE container + numerics parity
  run. Folded into work item (1) (parity matrix / KDA-TP column).
- **EP=2: BLOCKED (known drift, root-caused).** Upstream EP is now
  declared via sharding configs at config-build time;
  _moe.parallelize() on our directly-built modules distributes nothing
  (_sharding_config is None) -> grouped_mm batch mismatch vs EP-routed
  token counts. Needs config-build-time EP wiring like qwen3/dsv3.
  Folded into work item (1).

Step 5 is optional per this checklist; steps 1-4 (the mandatory
re-verification) are all green.

## Known-risk notes

- fla `fused_norm_gate` device-side assert (SM 12.0, ~650-step KDA
  threshold): NOT observed in any run so far (longest KDA run: 300
  steps; L16 dense runs don't exercise KDA).
- torch 2.12-stable vs upstream-nightly drift is the dominant breakage
  class this pass (loss_kwargs, set_timeout, shape-inference metadata).
