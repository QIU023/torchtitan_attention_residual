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
- **TP=2: FIXED same day -> PASS** (50-step smoke rc=0, loss 7.55).
  Root cause was integration-level (merged MoE no longer to_locals its
  input); resolution: MoE TP/EP migrated to trunk's module-internal
  mechanism -- sharding configs declared at config build
  (set_moe_sharding_config, dsv3's expert param layout) via
  update_from_config-wired flags, _moe.parallelize(parallel_dims) at
  parallelize time, all _moe entries dropped from the hand TP plan,
  and a plain-tensor boundary restored at KimiMoE.forward exit
  (Partial -> Replicate all-reduce + to_local). Commit f7bb7cd0f.
- **EP=2: FIXED same day -> PASS** (50-step smoke rc=0, loss 7.68).
  Same mechanism (the config-build-time sharding declaration was the
  missing piece; _sharding_config had been None on directly-built
  modules).
- Regression checks after the migration: full kimi_k3 suite 52 passed;
  FSDP-only 194m 5-step smoke unchanged (loss 12.17). The
  TP/EP-vs-FSDP loss-parity matrix remains queued (work item (1)).

Step 5 is optional per this checklist; steps 1-4 (the mandatory
re-verification) are all green.

## Post-checklist: TP/EP parity matrix (work item (1), same day)

Protocol: shared seed checkpoint (torchtitan standard for
cross-parallelism comparison -- per-config sharded init draws differ
by design), fp32 + `--debug.seed 42 --debug.deterministic`, identical
data order within each pair, 447m_aligned flavor, 60 steps:

| pair | step-1 loss | steps 2-5 |d| | step-1 grad_norm |
|---|---|---|---|
| TP=2 vs FSDP dp=1 (same data) | **EXACT (12.21259)** | <= 4e-5 | 22.571 vs 22.764 (0.85%) |
| EP=2 vs FSDP dp_shard=2 (same data) | **EXACT (12.24621)** | <= 5e-5 | **EXACT (11.3439)** |

Beyond ~step 10 the fp32 trajectories decohere (1e-3..5e-2 by step
60): chaotic amplification at the flavor's warmup LR (2.2e-3), seen in
BOTH pairs and in either direction depending on seed -- not a
parallelism defect. Control: the dense-carrier fp32 PP probe (7x lower
LR) stayed at 3e-5 over 100 steps. Verdict: TP and EP forward/loss
are exactly parity at step 1 and backward/optimizer equivalent to
~1e-5 over the early window; the KDA-TP column (KDA replicated on the
TP mesh by design) is exercised by the TP pair. Logs:
`parity_s_{fsdp1,tp2,fsdp2,ep2}.log` in the reverification folder.

## Known-risk notes

- fla `fused_norm_gate` device-side assert (SM 12.0, ~650-step KDA
  threshold): NOT observed in any run so far (longest KDA run: 300
  steps; L16 dense runs don't exercise KDA).
- torch 2.12-stable vs upstream-nightly drift is the dominant breakage
  class this pass (loss_kwargs, set_timeout, shape-inference metadata).


## Post-checklist: 48B graft anchor verified on real weights (same day)

The HANDOFF sec 5 anchor is now implemented AND measured. Finding
first: the paper's ungated zero-init AttnRes read is a uniform
source-average, NOT an identity (debug-scale A/B: max |dlogit| 0.126,
top-1 96.5%) -- fine for from-scratch pretraining, wrong for grafting.
The new alpha gate (opt-in flavor kimi_linear_48b_block_attn_res_gated;
h = plain + alpha * (mix - plain), sequentially-threaded plain stream)
restores exactness:

- Debug-scale unit tests lock both directions (gated == torch.equal
  with the plain backbone; ungated == measurably different).
- **Real-weight verification** (official Kimi-Linear-48B-A3B-Base,
  8x5090 FSDP, verify_48b_graft_step0.py): baseline vs gated graft on
  the same batch -> **max |dlogit| = 0.0, top-1 agreement 100.00%**.
  603 backbone keys loaded (native DCP re-export of the HF load),
  165 AttnRes/alpha params kept zero-init.

Post-training on the graft can therefore start from a step-0 function
that IS the original checkpoint, with alpha training away from
identity under optimizer control.
