# PR #13b — fla `causal_conv1d_{fwd,bwd}_kernel` device-side assert on Blackwell sm_120

## Status

🟡 **Patch written + applied locally via vendored shadow (no site-packages edit). Minimal repro script written but NOT YET RUN (all 8 GPUs busy with a live training run — a GPU repro would OOM/corrupt it). Bench-validation + upstream filing pending a free GPU.**

- **Ready patch diff**: [`causal_conv1d_PR13b.patch`](./causal_conv1d_PR13b.patch) — 2 hunks (fla 0.5.0), unified diff against `fla/modules/conv/triton/kernels.py`. `git apply`-able upstream.
- **Local application (this repo, 2026-05-30)**: vendored a patched copy of `kernels.py`; the shadow MetaPathFinder (same mechanism as PR13) maps the installed module to the vendored file. Installed site-packages fla stays pristine.
  - `phase5_vlm_multimodal_sft/vendored_fla/conv_triton_kernels_patched.py` (patched)
  - `phase5_vlm_multimodal_sft/vendored_fla/conv_triton_kernels_ORIG.py` (pristine, for diffing)
  - `phase5_vlm_multimodal_sft/vendored_fla/conv_triton_ops_ORIG.py` / `conv_triton_ops_patched.py` (pristine copies, IDENTICAL — `ops.py` needs NO change; kept only for completeness/diffing)
  - **Module to shadow** (wire into `sitecustomize.py` MetaPathFinder): `fla.modules.conv.triton.kernels`. `fla.modules.conv.triton.ops` does NOT need shadowing — the fix is entirely in `kernels.py`. (`ops.py` imports the kernel objects from `.kernels`, so shadowing `kernels` is sufficient and the autotune decorators live there.)

## Symptom (our path)

Kimi-Linear LM (KDA linear attention + MLA) trained on torchtitan, RTX 5090 (Blackwell sm_120, 170 SMs), Triton 3.6.0, fla 0.5.0, bf16. KDA uses a short convolution (`kda_short_conv_kernel_size=4`). Recurring device-side assert at roughly every ~50 training steps; crash frequency scales with sequence length (more frequent at seq_len 1536 than 1024). Traceback:

```
fla/modules/conv/short_conv.py:187   forward
fla/modules/conv/causal_conv1d.py:87 causal_conv1d
fla/modules/conv/triton/ops.py:367   forward          (CausalConv1dFunction.forward)
fla/modules/conv/triton/ops.py:54    causal_conv1d_fwd (kernel launch)
RuntimeError: Triton Error [CUDA]: device-side assert triggered
```

CUDA errors are asynchronous, so a surfacing line need not be the corrupting kernel. But here the launch site is also where Triton runs the **autotuner benchmarking loop synchronously** (on the first call for each new key tuple), so this site is a genuine offender — not merely an async artifact of a prior `fused_norm_gate` bwd (that was PR13's case).

## Bug location (fla 0.5.0)

```
fla/modules/conv/triton/kernels.py:
  L32   key=['D', 'W', 'NB']   # causal_conv1d_fwd_kernel autotuner
                      ^^^^ NB in autotuner key, never used in kernel body (L63-133)
  L149  key=['D', 'W', 'NB']   # causal_conv1d_bwd_kernel autotuner
                      ^^^^ same phantom key, never used in kernel body (L186-316)

fla/modules/conv/triton/ops.py:
  L49   NB = triton.cdiv(B*T, 1024)   # fwd launcher: computed, passed as constexpr
  L156  NB = triton.cdiv(B*T, 1024)   # bwd launcher: same
```

`NB` is computed in `ops.py` and passed in as a `tl.constexpr` kernel arg, but **never referenced in either kernel body** (verified by grep over L63-133 and L186-316 — zero hits).

## Root cause

This is **pattern #1 from PR #796 / PR13 (phantom autotuner key)** — and ONLY pattern #1.

- `NB = cdiv(B*T, 1024)` is a function of the total token count `B*T`. With variable/packed sequence lengths (1024 vs 1536, varlen batches, etc.), `B*T` lands in different `cdiv(_,1024)` buckets, so the autotune key tuple `(D, W, NB)` keeps changing.
- Each new key tuple forces Triton's `@triton.autotune` to **re-run its benchmarking loop** (timing every `BD ∈ {16,32,64,128} × num_warps ∈ {4,8,16,32}` config = 16 variants). On Blackwell sm_120 + Triton 3.6.0 this benchmarking itself crashes for certain variants at large grids — the documented #796 failure mode.
- Because `NB` is never used in the body, the re-autotuning buys nothing: the compiled kernel for a given `(D, W)` is identical regardless of `NB`. Removing `NB` from the key collapses the keyspace to `(D, W)` (constant within a model), so the autotuner runs **once** and is never re-triggered by T variation.

This matches every observed symptom: recurring (~each new T value), more frequent at larger seq_len (more distinct `NB` buckets exercised over a packed-batch run), and surfacing at the conv fwd launch (where autotune benchmarking runs synchronously).

### What we ruled out (pattern #2 from #796 does NOT apply here)

- **No grid-cap / BS<BT overlap.** `layernorm.py`/`fused_norm_gate.py` derive program count from `NS = min(get_multiprocessor_count(...), T)` and split `BS = cdiv(T, NS)`, which can make `BS < BT` and overlap `make_block_ptr` blocks. The conv launcher does NOT do this: grep for `get_multiprocessor_count` / `num_programs` in `ops.py` returns nothing. The grid is `(cdiv(D, meta['BD']), NT, B)` with `NT = cdiv(T, BT)` and **`BT` fixed at 64** (only `BD` is autotuned). Tiling is a clean disjoint partition of `(D, T, B)` — no SM-count-derived sharing, no adjacent-block overlap. Ruled out.
- **No `tl.static_assert` / `tl.device_assert`** in either kernel (grep: none). The "device-side assert" is the generic Triton wrapper for the autotuner-launch illegal access, not a user assert.
- **Initial-state masking path (`USE_INITIAL_STATE`, fwd L102-119 / bwd L255-293) is not the trigger.** In LM pretraining the short conv is called without an initial state (`cache=None`), so `USE_INITIAL_STATE=False` and the masked-load branch at L115 (`initial_state + ... + (o_x + W)[:, None]`) is never compiled into the hot path. The indexing there is also guarded by `m_c`/`m_x` masks. Not the recurring crash. (If a future decode/cache path exercises it, that is a separate investigation.)

So the fix is exactly the same shape as PR13's first two hunks (drop phantom `NB` from the key), with NO third hunk (no `NS`/`_MAX_BT` cap needed — that bug structurally cannot occur here).

## Proposed fix (mirrors PR #796 / PR13)

```diff
--- a/fla/modules/conv/triton/kernels.py
+++ b/fla/modules/conv/triton/kernels.py
@@ -29,7 +29,7 @@  # causal_conv1d_fwd_kernel autotuner
         for BD in [16, 32, 64, 128]
         for num_warps in NUM_WARPS_AUTOTUNE
     ],
-    key=['D', 'W', 'NB'],
+    key=['D', 'W'],   # PR13b: drop phantom NB (constexpr never used in body)
     **autotune_cache_kwargs,
 )
@@ -146,7 +146,7 @@  # causal_conv1d_bwd_kernel autotuner
         for BD in [16, 32, 64, 128]
         for num_warps in [4, 8, 16, 32]
     ],
-    key=['D', 'W', 'NB'],
+    key=['D', 'W'],   # PR13b: drop phantom NB (constexpr never used in body)
     **autotune_cache_kwargs,
 )
```

`NB` remains computed in `ops.py` and passed to the kernel (harmless constexpr; leaving the call sites untouched keeps the diff minimal and the kernel signature stable). The only change is removing it from the two `key=[...]` lists. See `causal_conv1d_PR13b.patch` for the committed version with full inline comments.

## Minimal repro SCRIPT — ⚠️ NOT YET RUN (GPUs busy)

```python
# repro_causal_conv1d_sm120.py
# Run on a single free RTX 5090 / B200 (sm_120) with Triton 3.6.0, fla 0.5.0.
# Expected on STOCK fla: "RuntimeError: Triton Error [CUDA]: device-side assert
#   triggered" / illegal memory access once a second distinct NB bucket is hit.
# Expected on PATCHED fla: runs clean across all T values (autotune runs once).
import torch
from fla.modules.conv.short_conv import ShortConvolution

D, W = 1024, 4          # KDA short conv: hidden_size 1024, kernel_size 4
conv = ShortConvolution(hidden_size=D, kernel_size=W, activation='silu').cuda().to(torch.bfloat16)

# Sweep T so B*T crosses several cdiv(_,1024) buckets -> forces re-autotune per T
# on stock fla. Mirrors packed/varlen batches at seq_len 1024 vs 1536.
for T in [1024, 1280, 1536, 2048, 1024, 1536]:
    x = torch.randn(2, T, D, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    y, _ = conv(x)                 # fwd: autotuner benchmarks on each NEW (D,W,NB)
    y.sum().backward()             # bwd: same phantom-NB autotune path
    torch.cuda.synchronize()       # surface the async device-side assert here
    print(f"T={T} OK, NB={ (2*T + 1023)//1024 }")
print("PASSED: no device-side assert across all T")
```

Validation plan once a GPU frees up:
1. Run the script against STOCK fla → expect the device-side assert (confirms repro).
2. Run against PATCHED fla (PYTHONPATH-shadowed) → expect clean PASS across all T.
3. Re-run Kimi-Linear pretrain @ seq_len 1536 for 8000+ steps crash-free.
4. Upstream test suite: `pytest tests/modules/test_conv.py` (or fla's conv tests).

## Workaround for users

Until merged:
1. Patch installed `kernels.py` per the diff above (remove `NB` from both `key` lists).
2. Or shadow `fla.modules.conv.triton.kernels` with a vendored patched copy via a `sitecustomize.py` MetaPathFinder (our approach — no site-packages edit).
3. Pin sequence length so `B*T` never crosses a `cdiv(_,1024)` boundary (avoids re-autotune) — usually infeasible with packed/varlen data.

## Why upstream

Affects every fla model that uses the short-conv / `causal_conv1d` Triton backend on Blackwell hardware with variable sequence lengths:
- Kimi Linear (KDA short conv, kernel_size 4)
- Mamba/Mamba2, DeltaNet (same `causal_conv1d` powers them)
- Any model packing variable-length batches on B200/B300/RTX 5090 with Triton ≥ 3.6.

Same precedent and fix shape as #796 (`layernorm.py`) and our PR13 (`fused_norm_gate.py`), now applied to the third sibling that carries the identical phantom-key bug: `conv/triton/kernels.py`.

## What we are NOT filing yet

The PR commit itself. Need a free GPU to: (1) run the repro on stock fla to confirm, (2) confirm patched fla passes the sweep + a clean multi-thousand-step pretrain, (3) run fla's conv test suite. Then file the issue + PR referencing #796 and our PR13 as precedent.
