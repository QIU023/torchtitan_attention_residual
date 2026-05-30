# PR #13c — fla norm-family Triton kernels: Blackwell sm_120 audit + l2norm fix

## Status

🟡 **l2norm patch written + vendored (no site-packages edit). NOT GPU-validated (all 8 GPUs busy with a live training run). Static diagnosis only.**

Continuation of the PR13 line:
- **PR13** — `fused_norm_gate.py` (phantom `NB` key + `BS<BT` grid overlap). Fixed + vendored.
- **PR13b** — `causal_conv1d` sm_120.
- **PR13c (this)** — audit of the remaining autotuned norm-family modules
  (`l2norm`, `layernorm`, `token_shift`, `rotary`, `fused_bitlinear`) for the
  same two root causes. Only **`l2norm`** has a confirmed bug that is **in our
  KDA+MLA path**, so only `l2norm` is patched.

Env: 8× RTX 5090, Blackwell **sm_120**, 170 SMs, Triton **3.6.0**, fla **0.5.0**
at `/usr/local/lib/python3.12/dist-packages/fla/`, bf16.

## The bug class (mirrors upstream fla #796)

1. **Phantom autotuner key** — a constexpr (`NB`) listed in
   `@triton.autotune(key=[...])` but never referenced in the kernel body. It
   only mutates the autotune cache key, forcing re-autotuning each time `T`
   crosses an `NB` boundary; on sm_120 the autotuner itself crashes
   (device-side assert) benchmarking variants at large grids.
2. **Grid-cap / block overlap** — `NS = min(get_multiprocessor_count(...), T)`,
   `BS = ceil(T/NS)`. When `BS < BT` (largest autotuned block-tile), adjacent
   programs' `make_block_ptr` blocks overlap on an output tensor → illegal
   memory access.

A module is only affected by #2 if it (a) caps the grid with
`get_multiprocessor_count` AND (b) writes outputs via `make_block_ptr` block
tiles whose offsets can overlap. A plain `cdiv(T, BT)` grid (one program per
tile) cannot overlap and is immune to #2.

---

## Per-module findings

### `fla.modules.l2norm` — BUG #1 only — IN PATH — **PATCHED**

- **Symptom**: device-side assert / autotuner crash on sm_120 when `T` sweeps
  several `NB = cdiv(T, 2048*32)` ranges (long-context or accumulated-token
  training). Surfaces as a `Triton Error [CUDA]: device-side assert triggered`,
  possibly async (reported at a later CUDA op).
- **Exact bug lines (fla 0.5.0)**:
  - `l2norm.py:78` — `key=["D", "NB"]` on `l2norm_fwd_kernel`. `NB` is a
    `tl.constexpr` param (`l2norm.py:90`) **never used** in the kernel body
    (L93–103).
  - `l2norm.py:108` — `key=["D", "NB"]` on `l2norm_bwd_kernel`. Same:
    `NB` declared (L121) but unused (L124–134).
- **NO bug #2**: the launchers use `grid(meta) = (cdiv(T, meta["BT"]),)`
  (`l2norm.py:165`, `l2norm.py:215`) — one program per tile, no
  `get_multiprocessor_count`, no `NS`/`BS` sharing. Adjacent programs write
  disjoint `make_block_ptr` blocks at offset `i_t*BT`. So **only the NB key is
  removed**; no NS cap is added (would be a no-op here).
- **In-path proof**: `fla/ops/kda/chunk.py:12` `from fla.modules.l2norm import
  l2norm_bwd, l2norm_fwd`; called at `chunk.py:53-54` (`l2norm_fwd(q)` /
  `l2norm_fwd(k)`) and `chunk.py:144-145` (bwd) for q/k normalization when
  `use_qk_l2norm_in_kernel=True`. Our Kimi-Linear model
  (`torchtitan/.../kimi_linear/model.py:569`) calls `chunk_kda(...,
  use_qk_l2norm_in_kernel=True)`, so the **chunk training path hits l2norm
  every KDA layer every step**.
- **Fix**: drop `"NB"` from both `key=[...]` lists → `key=["D"]`. 2 hunks. See
  `l2norm.patch`. Vendored: `vendored_fla/l2norm_patched.py` (+ `l2norm_ORIG.py`).
- **Repro**: `repro_l2norm.py` (NOT RUN).

### `fla.modules.layernorm` — REFERENCE (#796 fix already applied) — NOT IN PATH — skip

- This is the upstream #796-fixed module and serves as the reference for the
  whole class. Bug #2 is **mitigated**: the bwd launcher caps
  `NS = min(cdiv(get_mp_count, G), T//G) * G` (`layernorm.py:660`) and the
  kernel masks tail rows with `m_t = (i_t + arange(BT)) < min(i_sg*BS+BS, Tg)`
  (`layernorm.py:416-421`) plus the `Tg`/group-strided `make_block_ptr`
  (L386–398). These are the #796 markers — present here, absent in the stock
  `fused_norm_gate.py`/`l2norm.py`, which is exactly why PR13 + PR13c are needed.
- Note: `"NB"` is **still listed** in this module's keys (`layernorm.py:190,
  337`) and `NB` is still passed (now `cdiv(T, 2048)`, L577/L669). Upstream #796
  left the phantom-key half in place for the un-gated variant; it is a latent
  re-autotune cost but the `make_block_ptr` overlap (the crashing half) is
  fixed. We do **not** touch `layernorm.py` because our model uses torch-native
  `nn.RMSNorm` for all plain norms (`kimi_linear/model.py:299,754,757,803`,
  etc.), never `fla.modules.layernorm` — out of path.

### `fla.modules.token_shift` — BUG #1 present — NOT IN PATH — skip (note only)

- `token_shift.py:156` and `:308` — `key=['BD', 'NB']` on
  `token_shift_fwd_kernel_long` / `token_shift_bwd_kernel_long`. `NB`
  (`= cdiv(B*T, 1024)`, L421/L477) is a `tl.constexpr` (L171/L323) **never used**
  in the kernel body. So bug #1 is present.
- **NO bug #2**: the long kernels write with plain pointer offsets
  (`x + offset`, masked by `m_d`), not overlapping `make_block_ptr` block tiles;
  grid is `(cdiv(D,BD), NT, N)`. The `get_multiprocessor_count` use (L415/L473)
  only sizes `BT`, it is not an `NS`/`BS` overlap cap.
- **Not in path**: token-shift is an RWKV-family op. Our Kimi-Linear stack
  (KDA + MLA) never imports `fla.modules.token_shift` (only the package
  `modules/__init__.py` re-exports it). Bug present but not in our path → not
  patched. If an RWKV/HGRN model is ever trained on this box, mirror the
  l2norm NB-key removal here.

### `fla.modules.rotary` — NO bug — NOT IN PATH — skip

- `rotary.py:43` — `key=['B', 'H', 'D', 'INTERLEAVED']`. **No phantom `NB`**
  (no `NB` param at all). Bug #1 absent.
- Grid `(NT, B, H)` (`rotary.py:191`); `get_multiprocessor_count` only sizes
  `BT` (`rotary.py:186`); all stores are pointer+`mask` (L109-110), masked by
  `m_t`/`m_d`. No overlapping `make_block_ptr` block-tile writes → bug #2 absent.
- **Not in path** anyway: Kimi-Linear's MLA RoPE is implemented natively
  (`torchtitan.models.common.rope`), not via `fla.modules.rotary`. No change.

### `fla.modules.fused_bitlinear` — NO matching bug — NOT IN PATH — skip (note only)

- `fused_bitlinear.py:69, 204` — `key=["N", "HAS_RESIDUAL", ...]` / `["N",
  "HAS_DRESIDUAL", ...]`. **No phantom `NB`**. Bug #1 absent.
- The bwd launcher uses `sm_count = get_multiprocessor_count` with
  `rows_per_program = ceil(M / sm_count)` (`fused_bitlinear.py:344-348`), but
  the kernel is a per-row strided loop with `cols < N` masks, not the
  `make_block_ptr` `BS<BT` block-tile pattern. Structurally NOT the #796 bug.
- **Not in path**: bitlinear is for BitNet-style models only; our stack has no
  `FusedBitLinear`/`BitLinear`. No change.

---

## Modules to shadow (sitecustomize MetaPathFinder)

Add to the existing vendored-fla shim the same way `fla.modules.fused_norm_gate`
is shadowed (PR13). For PR13c, shadow exactly:

- `fla.modules.l2norm`  → `vendored_fla/l2norm_patched.py`

(That is the only PR13c module that is both confirmed-buggy and in our path.
`layernorm`, `token_shift`, `rotary`, `fused_bitlinear` are intentionally NOT
shadowed.) The shim is a one-target `_TARGET` per finder; either add a second
`MetaPathFinder` for `fla.modules.l2norm` or generalize the existing finder to a
`{target: patched_path}` dict — both keep site-packages pristine.

## Upstream filing

Mirror PR #796 / our PR13. For `l2norm.py`: remove `"NB"` from the two
`@triton.autotune(key=...)` lists (no NS cap needed — l2norm's grid is
`cdiv(T, BT)`, immune to the overlap half). Optionally fold in the
`token_shift.py` NB-key removal as a sibling cleanup, noting it is latent
(bug #1 only, also no overlap).

## What is NOT done

- No GPU validation (all 8 GPUs busy). The l2norm patch is diagnosed statically;
  `git apply --check` passes against the installed `fla/modules/l2norm.py`.
- No upstream issue/PR filed yet — pending a green local KDA run with the shim
  wired in.
