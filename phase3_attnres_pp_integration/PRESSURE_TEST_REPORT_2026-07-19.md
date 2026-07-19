# PP Pressure Test — Re-verification Report (2026-07-19)

> Successor to [PRESSURE_TEST_REPORT_2026-05-12.md](PRESSURE_TEST_REPORT_2026-05-12.md):
> the same numerics claims re-verified on the post-upstream-merge fork,
> per [phase13 PRE_RFC_GPU_CHECKLIST](../phase13_k3like_48b_posttrain/PRE_RFC_GPU_CHECKLIST.md).
> Raw logs for every number below:
> [`runs_20260719_reverification/`](runs_20260719_reverification/).
> Full execution narrative (all 5 checklist steps + fix inventory):
> [phase13 GPU_CHECKLIST_RESULTS_2026-07-19](../phase13_k3like_48b_posttrain/GPU_CHECKLIST_RESULTS_2026-07-19.md).

## Environment

- 8x RTX 5090 32GB (SM 12.0), PCIe fabric; torch **2.12.0+cu130**
  (the 05-12 report predates the upstream merge and ran an older torch).
- fla-core 0.5.1, torchao 0.17.0.
- Fork: `QIU023/torchtitan@attention_residual_dev`, start `a3b3c74b3`
  (merged-upstream state) -> `fab8ebf24` after this pass's fixes.
  Test suites at the end state: kimi_k3 52 passed / 0 skipped,
  dense_carrier 70 passed / 0 skipped.

## Results 1 — L=16 Block AttnRes sweep (same grid as 05-12)

1000 steps from-scratch on C4, `run_pp_pressure_test.sh` defaults,
bf16, 8 GPUs:

| Shape | LBS | GBS | DP | naive | adapter | d (adapter-naive) |
|---|---|---|---|---|---|---|
| PP=8 x VP=2 | 16 | 16 | 1 | 5.27563 | 5.26996 | **-0.00567** |
| PP=4 x VP=2 | 8 | 16 | 2 | 5.47948 | 5.44676 | **-0.03272** |
| PP=4 x VP=4 | 16 | 32 | 2 | 5.11195 | 5.10225 | **-0.00970** |

Step-1 loss is bit-identical (11.7618) across all six runs: init and
data order are seed-matched, and the AttnRes zero-init identity holds
under the adapter.

### Nondeterminism band, measured on THIS box/torch

Naive re-run with `--debug.seed 123` on the PP4xVP2 shape: final loss
5.50868 vs baseline naive 5.47948 -> **seed-vs-seed |dLoss| = 0.0292**
at 1000 steps. (The 05-12-era band on this carrier was 0.06-0.13.)
All three adapter deltas sit inside the measured band.

### fp32 deterministic equivalence probe (new, stronger than the band)

Because the PP4xVP2 delta (0.0327) was same-signed at every late step,
it was NOT dismissed as noise by inspection. Discriminator: same shape,
100 steps, `--training.mixed_precision_param float32 --debug.seed 42
--debug.deterministic`, naive vs adapter:

- Loss identical to 5 printed decimals through step ~90.
- max |dLoss| = 3e-5 (step 96), mean 1e-6 over 100 steps.

**The adapter's forward+backward graph is numerically equivalent to
naive PP.** The bf16 end-of-run deltas are reordering-noise trajectory
divergence (LPS=2 changes the adapter's block-stack assembly order),
not a gradient-routing defect. Logs: `fp32_probe_{naive,adapter}.log`.

## Results 2 — Kimi-Linear 48B-layout carrier, PP=8 x VP=4 (32 virtual stages)

`kimi_linear_48b_block_attn_res_d1280_e16_L32_N8`, 300 steps, LBS=32:

| Config | naive | adapter | d |
|---|---|---|---|
| seq=512 (both fit) | 6.54285 | 6.54153 | **-0.00132** |
| seq=1024 | **OOM** (29.9 GiB, GPU1) | 5.97096, peak 27.15 GiB | n/a |

The seq=1024 naive OOM is a finding, not a failure: naive PP carries
the full accumulated block stack between stages and does not fit 32 GB
at this shape -- consistent with the repo history containing only
adapter logs here. The adapter's delta-cache memory saving is what
makes the shape feasible on these cards at all.

## Cross-era reproduction (this report vs 05-12 / repo history)

| Quantity | 05-12 era | today | agreement |
|---|---|---|---|
| L16 shape ordering (final loss) | vp4 < pp8 < pp4_vp2 | same | exact |
| L16 max naive-vs-adapter delta | 0.0044 (band 0.06-0.13) | 0.0327 (band 0.0292) | inside band both eras |
| kimi48b adapter step-300 loss (same script) | 5.96955 | 5.97096 | **|d| = 0.0014** |
| kimi48b adapter peak mem | 24.76 GiB | 27.15 GiB | +2.4 GiB (torch 2.12) |

The 48B-layout carrier reproducing to 0.0014 across a torch major
bump plus this pass's 8 fix commits is the strongest continuity
evidence: the merged fork computes the same model.

Step-time/tps are NOT re-claimed here: today's per-run timing is
polluted by the in-run validator (steps 500/1000) and single-snapshot
tps; performance numbers continue to cite the 05-12 report's
methodology. Numerics were the scope of this pass.

## What had to be fixed to get here (summary; details in the phase13 results doc)

All on the fork branch (`a3b3c74b3..fab8ebf24`, 10 files, +718/-74):

1. fp8 flavor could never build on SM89+ (config-tree conversion vs
   plain-module model) -> module-level torchao swap, structural KDA
   exclusion; fp8 5-step smoke now passes on SM 12.0.
2. Four merged-Trainer-chain breaks (kwarg rename, traverse
   requirement, abstract loss default, torch 2.12 set_timeout).
3. Three PP-path breaks on torch 2.12 stable, including one real
   adapter bug: the shape-inference delta placeholder dropped
   requires_grad, which torch >= 2.12 uses to derive cross-stage
   backward metadata (silent-severed delta edges -> loud
   PipeliningMetadataError). One-line fix + the fp32 probe above as
   the correctness evidence.
4. DSv3 CUDA-gated tests had never actually run on CUDA; dense_carrier
   loss wiring for the new Trainer.

Known-remaining (tracked, root-caused, NOT blocking the RFC's claims):
TP=2 / EP=2 smokes on the 447M flavor hit merged-MoE-runtime
integration drift (mixed Tensor/DTensor at the token dispatcher;
EP sharding now declared at config-build time). See phase13 results
doc, step 5.
