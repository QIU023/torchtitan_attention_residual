# hot_patches/fla_fused_norm_gate_sm120_kda/

## What

Patch `fla/modules/fused_norm_gate.py` (fla-core 0.5.0) to fix the same
class of bug that PR #796 fixed in `fla/modules/layernorm.py` but not in
the gated sibling.

Three changes:

1. **L31** — `layer_norm_gated_fwd_kernel` autotune key: drop the `"NB"`
   key. `NB` is computed by the launcher (`cdiv(T, 2048 * 32)`) but never
   referenced in the kernel body; on Blackwell sm_120 + Triton 3.6.0 the
   autotuner re-runs per new T range and crashes when benchmarking the
   `HAS_DRESIDUAL=False` variant at large grid sizes.
2. **L198** — `layer_norm_gated_bwd_kernel` autotune key: drop `"NB"`.
3. **L562-565** — `layer_norm_gated_bwd` launcher: cap `NS` so
   `BS >= max_BT (= 64)` to prevent adjacent programs' `make_block_ptr`
   blocks from overlapping on the output `dx`. Original code used
   `NS = min(SM_count, T)` and `BS = ceil(T / NS)`, which gives `BS < BT`
   when `SM_count > T / max_BT` — observed on RTX 5090 (170 SMs) with our
   stage 2 SFT input shapes (T=4640).

Diff in this dir:
  - `original_fla_0.5.0_fused_norm_gate.py` ← bit-identical backup
  - `patched_fused_norm_gate.py` ← what's currently deployed

## Why

Without this patch, every Kimi-Linear SFT run on RTX 5090 (Blackwell
sm_120) crashes deterministically at ~step 2500 with
`Triton Error [CUDA]: device-side assert triggered`. The crash surfaces
in the *forward* call to `rms_norm_gated` because CUDA reports async
errors at the next kernel launch — the actual fault is in the bwd
kernel from the prior microbatch.

See `../../Raising_PRs/PR13_fla_fused_norm_gate_sm120_kda_crash/PR.md`
for the upstream issue body + repro.

## When applied

2026-05-17 20:02 UTC — applied live to
`/usr/local/lib/python3.12/dist-packages/fla/modules/fused_norm_gate.py`.

## How to apply / rollback

```bash
# Apply (idempotent — verifies md5 of backup first):
bash apply.sh

# Rollback to upstream 0.5.0:
bash rollback.sh
```

## Validation plan

1. Next stage 2 retry (after current attempt's KDA crash) picks up the
   patched code via fresh import.
2. If next two retries also crash at the same ~step 2500 window, the
   patch did not address the root cause — rollback and investigate.
3. If the next retry runs past step 5000 without an assert, the patch
   works; let stage 2 train through.

## Upstream tracker

- Mirror of [PR #796](https://github.com/fla-org/flash-linear-attention/pull/796)
  applied to the gated variant.
- Our PR draft: `../../Raising_PRs/PR13_fla_fused_norm_gate_sm120_kda_crash/`
- Once upstream lands the fix in `fused_norm_gate.py` (or fla bumps a
  release with the fix), pip install will overwrite the patch and
  `apply.sh` will fail md5 — at which point delete this dir.
