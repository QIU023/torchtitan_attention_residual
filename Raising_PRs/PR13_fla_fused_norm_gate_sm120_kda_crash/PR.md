# PR #13 — fla `fused_norm_gate.py` device-side assert on Blackwell sm_120

## Status

🟡 **Patch written + applied locally via vendored shadow (no site-packages edit); bench-validating on seq-KD S5 (seq_len 1536). Upstream filing still pending green local run.**

- **Ready patch diff**: [`fused_norm_gate_PR13.patch`](./fused_norm_gate_PR13.patch) — 3 hunks (fla 0.5.0), unified diff against `fla/modules/fused_norm_gate.py`. `git apply`-able upstream.
- **Local application (this repo, 2026-05-29)**: vendored a patched copy + a `sitecustomize.py` MetaPathFinder that shadows `fla.modules.fused_norm_gate` at import — installed site-packages fla stays pristine.
  - `phase5_vlm_multimodal_sft/vendored_fla/fused_norm_gate_patched.py` (patched)
  - `phase5_vlm_multimodal_sft/vendored_fla/fused_norm_gate_ORIG.py` (pristine, for diffing)
  - `phase5_vlm_multimodal_sft/vendored_fla/sitecustomize.py` (the shadow finder)
  - Activated by `run_seqkd_sft_autoresume.sh` prepending `vendored_fla/` to `PYTHONPATH`.
  - Verified: `import fla.modules.fused_norm_gate` resolves to the vendored file; all autotuner keys NB-free; NS `_MAX_BT` cap present; `from fla.modules import FusedRMSNormGated` OK.
- **Why applied now**: seq-KD S5 (KDA SFT, distilled mix665k) hit this crash *far* more often at `seq_len=1536` than the historical ~step-2500 cadence — consistent with PR13's mechanism (larger T → more autotuner re-triggering). Crash traceback surfaced in `causal_conv1d_fwd`, but per the async-error note below the corrupting kernel is the `fused_norm_gate` bwd from a prior microbatch. Bench validation = whether S5 @ 1536 now runs crash-free.

## Where to file

- **Issue**: https://github.com/fla-org/flash-linear-attention/issues/new
- **PR**: against `main`, mirroring the closed/open fix in `layernorm.py` (PR #796) but applied to the sibling `fused_norm_gate.py`.

## Title

```
[Bug] layer_norm_gated_{fwd,bwd}_kernel: same NB-autotune-key + BS<BT crash as #796 layernorm fix, but in fused_norm_gate.py — Blackwell sm_120 + Triton 3.6.0
```

## Background

PR #796 ([Layernorm] Fix autotuner crash and OOB writes in layer_norm_bwd on high-SM GPUs) identifies two bugs in `fla/modules/layernorm.py` on Blackwell sm_120 (188 SMs B200, 170 SMs RTX 5090):

1. **Phantom `NB` in autotuner key** triggers re-autotuning per new T range; on sm_120 the autotuner itself crashes when benchmarking certain variants at large grid sizes.
2. **Overlapping writes in bwd** when `BS = cdiv(T, NS) < BT`, adjacent programs' `make_block_ptr` blocks overlap on the output `dx`, causing illegal memory accesses.

The fixes in #796 (remove `NB` from key + cap `NS` so `BS >= max_BT`) apply only to `layernorm.py`. The structurally identical bugs in `fla/modules/fused_norm_gate.py` are not addressed.

## Symptom (our path)

Kimi-Linear LM trained on torchtitan, RTX 5090 (Blackwell sm_120, 170 SMs), Triton 3.6.0, fla 0.5.0. KDA + MLA architecture; the MLA `o_norm` is `rms_norm_gated` from `fused_norm_gate`. Crash at ~step 2500 each run (per-rank), traceback:

```
File "fla/modules/fused_norm_gate.py", line 1042, in forward
  return rms_norm_gated(...)
File "fla/modules/fused_norm_gate.py", line 857, in rms_norm_gated
  return LayerNormGatedFunction.apply(...)
File "fla/modules/fused_norm_gate.py", line 659, in forward
  y, mean, rstd, residual_out = layer_norm_gated_fwd(...)
RuntimeError: Triton Error [CUDA]: device-side assert triggered
```

Stack points at fwd, but CUDA errors are asynchronous — the bwd kernel triggers the assert on a prior microbatch and the error surfaces at the next CUDA op (= next-iteration fwd of `o_norm`).

Same env as #796 reproducer (sm_120 + Triton 3.6.0), but the offending kernels are `layer_norm_gated_fwd_kernel` / `layer_norm_gated_bwd_kernel` in `fused_norm_gate.py`, not the un-gated variants in `layernorm.py`.

## Bug locations (fla 0.5.0)

```
fla/modules/fused_norm_gate.py:
  L31   key=["D", "NB", "IS_RMS_NORM", "STORE_RESIDUAL_OUT", "HAS_RESIDUAL", "HAS_WEIGHT"]
                ^^^^^ NB in autotuner key, never used in kernel body
  L198  key=["D", "NB", "IS_RMS_NORM", "HAS_DRESIDUAL", "HAS_WEIGHT"]
                ^^^^^ same in bwd kernel
  L536-538  NS = min(get_multiprocessor_count(x.device.index), T)
            BS = math.ceil(T / NS)
            # If NS > T // max_BT, then BS < max_BT and adjacent programs'
            # make_block_ptr blocks overlap on dx. Same root cause as #796.
```

Kernel selection: for D ≤ 512 (e.g. head_dim=64 in MLA), the multi-row tiled `layer_norm_gated_{fwd,bwd}_kernel` is used (the affected ones). For D > 512 the per-row `_kernel1` is used (no NB in key, no NS sharing).

## Minimal repro (mirrors #796 but for the gated variant)

```python
import torch
from fla.modules.fused_norm_gate import FusedRMSNormGated

# Use D=256 (mirror #796 reproducer) but the bug also triggers with D=64.
norm = FusedRMSNormGated(256).cuda().to(torch.bfloat16)

# T = 24000 hits multiple NB ranges; triggers re-autotuning.
T, D = 24000, 256
x = torch.randn(T, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)
g = torch.randn(T, D, dtype=torch.bfloat16, device="cuda", requires_grad=True)

# HAS_RESIDUAL=False, HAS_DRESIDUAL=False path:
norm(x, g).sum().backward()  # expected: CUDA error: illegal memory access
                              # observed on B200/5090 (sm_120) + Triton 3.6
```

Same repro succeeds when `D=512` (uses `_kernel1` per-row variant, no shared NS) — confirms the bug is in the tiled path's grid-cap interaction, not numerical.

## Proposed fix (mirrors PR #796 in `layernorm.py`)

```diff
--- a/fla/modules/fused_norm_gate.py
+++ b/fla/modules/fused_norm_gate.py
@@ -28,7 +28,7 @@
 @triton.autotune(
     configs=[triton.Config({"BT": BT}, num_warps=num_warps) for BT in [16, 32, 64] for num_warps in [4, 8, 16]],
-    key=["D", "NB", "IS_RMS_NORM", "STORE_RESIDUAL_OUT", "HAS_RESIDUAL", "HAS_WEIGHT"],
+    key=["D", "IS_RMS_NORM", "STORE_RESIDUAL_OUT", "HAS_RESIDUAL", "HAS_WEIGHT"],
     **autotune_cache_kwargs,
 )

@@ -195,7 +195,7 @@
 @triton.autotune(
     configs=[triton.Config({"BT": BT}, num_warps=num_warps) for BT in [16, 32, 64] for num_warps in [4, 8, 16]],
-    key=["D", "NB", "IS_RMS_NORM", "HAS_DRESIDUAL", "HAS_WEIGHT"],
+    key=["D", "IS_RMS_NORM", "HAS_DRESIDUAL", "HAS_WEIGHT"],
     **autotune_cache_kwargs,
 )

@@ -533,9 +533,15 @@  # in layer_norm_gated_bwd launcher
-    # cap program count to T so no program is completely idle.
-    NS = min(get_multiprocessor_count(x.device.index), T)
-    BS = math.ceil(T / NS)
+    # Cap NS so each program's BS >= max autotuned BT (=64). On high-SM
+    # GPUs with small T, BS < BT causes adjacent programs' make_block_ptr
+    # blocks to overlap on dx, corrupting GPU memory. Same root cause as
+    # PR #796 fix in layernorm.py.
+    _MAX_BT = 64  # largest BT in autotuner configs above
+    NS_cap_sm = get_multiprocessor_count(x.device.index)
+    NS_cap_overlap = max(T // _MAX_BT, 1)
+    NS = min(NS_cap_sm, NS_cap_overlap)
+    BS = math.ceil(T / NS)
```

Same pattern as #796 (different launcher signature → simpler since `fused_norm_gate` doesn't use the G-group structure of `layernorm.bwd`).

## Tests to add

Mirror #796's `test_rmsnorm_varying_nb_no_residual` and `_with_residual` but importing `FusedRMSNormGated` from `fla.modules.fused_norm_gate`. Parametrize over T ∈ {100, 500, 5000, 10000, 20000, 24000} × D ∈ {64, 256} to cover both kernel paths.

## Workaround for users

Until merged:
1. Use `D > 512` if possible (falls back to per-row kernel) — usually not feasible.
2. Patch installed `fused_norm_gate.py` locally per the diff above.
3. Cap `tl.program_id(0)` grid count by hand — possible but more invasive.

## Why this matters

Affects every fla model using `FusedRMSNormGated` on Blackwell hardware:
- Kimi Linear (KDA + MLA `o_norm`)
- DeepSeek V3 (gated MLA variant)
- Any model using `rms_norm_gated` on B200/B300/RTX 5090 with Triton ≥ 3.6.

In our case, KDA SFT on RTX 5090 crashes every ~2500 steps. Auto-retry (FSDP ckpt resume) absorbs the loss but wastes ~5 min boot per cycle.

## What we are NOT filing yet

The PR commit itself. Need to:
1. Validate the patch in our env (apply, re-run stage 2, confirm no crash for 8000+ steps).
2. Run the upstream test suite (`pytest tests/modules/test_fused_norm_gate.py`).
3. Optionally test the int64-overflow fix scope (PR #818 is open for the same kernels) — may want to bundle.

Once we have green CI locally + a clean stage 2 run, file the issue and open the PR with reference to #796 as the precedent.
