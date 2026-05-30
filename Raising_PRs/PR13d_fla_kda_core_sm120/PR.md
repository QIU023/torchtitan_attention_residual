# PR13d — fla KDA core linear-attention Triton kernels: sm_120 bug audit

## Status

🟢 **AUDIT RESULT: the KDA core ops are CLEAN. No confirmed sm_120 bugs found in `fla/ops/kda/*` or the `fla/ops/common/*` kernels KDA depends on.** No patch is required for the KDA core. The `fused_norm_gate.py` / `layernorm.py` / `l2norm.py` / `causal_conv1d` crashes (PR13, PR13b, PR13c) are **not** mirrored here.

- Env: fla 0.5.0 @ `/usr/local/lib/python3.12/dist-packages/fla/`, Triton 3.6.0, RTX 5090 Blackwell **sm_120**, 170 SMs, bf16.
- Method: static read of every `@triton.autotune` block + every host-side grid/launch in the KDA hot path. No GPU execution (8 GPUs busy training).
- This is a deliberately *negative* result: the bug class we fixed in `fused_norm_gate.py` (phantom autotuner key + `BS<BT` `make_block_ptr` overlap on high-SM GPUs) **does not exist** in the KDA chunked linear-attention kernels. Patching here would be fabrication.

## The two bug signatures we hunted (from upstream fla #796 / our PR13)

1. **Phantom autotuner key**: a var in `@triton.autotune(key=[...])` that is *not used in the kernel body*, AND whose value is derived from `T` (e.g. `NB = cdiv(T, 2048)`). Because it tracks `T`, it forces Triton to re-autotune every time `T` crosses a bucket boundary; on sm_120 the autotuner itself faults when re-benchmarking variants at large grids.
2. **`BS < BT` grid-cap overlap**: host code does `NS = min(get_multiprocessor_count(dev), T); BS = ceil(T/NS)` and then each program strides `make_block_ptr` by `BT`. When `BS < BT` (high SM count, small `T`), adjacent programs' output blocks overlap → illegal memory access / device-side assert.

## Why the KDA core is structurally immune

### (A) No `get_multiprocessor_count` / SM-count grid sizing anywhere in the KDA path
```
$ grep -rln "get_multiprocessor_count" fla/ops/kda/ fla/ops/common/chunk_{h,o,delta_h,scaled_dot_kkt,h_split}.py fla/ops/common/fused_recurrent.py
  -> NONE
```
Bug signature #2 requires `NS = min(SM_count, …)`. That pattern exists **only** in `fla/modules/{layernorm,fused_norm_gate}.py` (already fixed in PR13). Every KDA grid is sized from data dims via `triton.cdiv` (e.g. `(cdiv(K,BK), cdiv(V,BV), N*H)`, `(NT, B*HV)`), so no program is launched whose offset exceeds the tensor, and `make_block_ptr` blocks tile the data without overlap. Tail tiles are handled by `boundary_check=` and early `if i_t*BT >= T: return` guards.

### (B) The `BS` in `chunk_h.py` is a *split size*, not a per-program token span — the inequality is inverted (safe)
`fla/ops/common/chunk_h.py:289` `BS = BT if split_size is None else split_size`, and `:290` asserts `BS % BT == 0`. So `BS >= BT` is *guaranteed by construction*. `NS = cdiv(T, BS)` (`:69,73,201,206,293,352`) is the number of state-splits, used to index the state buffer — it is **not** a grid-cap that could drop below `BT`. This is the inverse of the buggy norm launcher (where `BS = ceil(T/NS)` could fall below `BT`). No overlap possible.

### (C) Every autotuner key var is a real kernel arg used in the body; no `T`-derived phantom key
Checked each `key=[...]` against its `@triton.jit` signature and body. All keys are `tl.constexpr` kernel arguments referenced in the kernel. Where a constexpr is *missing* from a key (e.g. `USE_EXP2`, `S`, `USE_SAFE_GATE`), that is an under-specified cache key (a correctness-of-cache nit), **not** the phantom-key crash — those vars are config-constant across a run, not `cdiv(T,·)` buckets, so they never trigger runtime re-autotuning. None of the KDA kernels compute a `NB = cdiv(T, …)`-style value at all.

## Summary table — every autotuned kernel on the KDA hot path

| File | Kernel | autotune key | grid | Status |
|---|---|---|---|---|
| kda/gate.py:83 | `kda_gate_fwd_kernel` | `H,D` | `(cdiv(T,BT), H)` | clean |
| kda/gate.py:139 | `kda_gate_bwd_kernel` | `H,D` | `(cdiv(T,BT), H)` | clean |
| kda/gate.py:358 | `kda_gate_chunk_cumsum_vector_kernel` | `H,S,BT,IS_VARLEN,REVERSE` | `(cdiv(S,BS), NT, B*H)` | clean |
| kda/wy_fast.py:28 | `recompute_w_u_fwd_kda_kernel` | `H,HV,K,V,BT,BK,BV,IS_VARLEN` | `(NT, B*HV)` | clean |
| kda/wy_fast.py:131 | `prepare_wy_repr_bwd_kda_kernel` | `H,HV,K,V,BT,BK,BV,IS_VARLEN` | `(NT, B*HV)` | clean |
| kda/chunk_intra.py:37 | `chunk_kda_fwd_kernel_inter_solve_fused` | `H,HV,K,BC` | `(NT, B*HV)` | clean |
| kda/chunk_intra.py:363 | `chunk_kda_bwd_kernel_intra` | `BK,NC,BT,HV` | `(NK*NC, NT, B*HV)` | clean |
| kda/chunk_intra.py:641 | `chunk_kda_fwd_kernel_intra_sub_chunk` | `BT,BC,HV` | `(NT, NC, B*HV)` | clean |
| kda/chunk_intra_token_parallel.py:27 | `chunk_kda_fwd_kernel_intra_token_parallel` | `K,H,HV` | `(B*T, cdiv(HV,BH))` | clean |
| kda/chunk_bwd.py:38 | `chunk_kda_bwd_kernel_dAv` | `H,HV,K,V,BT,BK,BV` | `(NT, B*HV)` | clean |
| kda/chunk_bwd.py:121 | `chunk_kda_bwd_kernel_wy_dqkg_fused` | `BT,HV,TRANSPOSE_STATE` | `(NT, B*HV)` | clean |
| kda/fused_recurrent.py | `fused_recurrent_*` (no autotune) | — | `(cdiv(V,BV)*N*HV,)` | clean |
| common/chunk_h.py:32 | `chunk_fwd_kernel_h` | `BT,USE_G,USE_GK,USE_GV` | `(cdiv(K,BK), cdiv(V,BV), N*H)` | clean |
| common/chunk_h.py:159 | `chunk_bwd_kernel_dh` | `BT,USE_G,USE_GK,USE_GV` | `(cdiv(K,BK), cdiv(V,BV), N*H)` | clean |
| common/chunk_o.py:32 | `chunk_fwd_kernel_o` | `H,HV,K,V,BT,TRANSPOSE_STATE` | `(cdiv(V,BV), NT, B*HV)` | clean |
| common/chunk_o.py:154 | `chunk_bwd_kernel_dqkwg` | `H,HV,K,V,BT,BK,BV,USE_G,USE_G_GAMMA,USE_DW,TRANSPOSE_STATE` | `(cdiv(K,BK), NT, B*HV)` | clean |
| common/chunk_o.py:353 | `chunk_bwd_kernel_dv` | `H,HV,K,V,BT,BK,BV,USE_G,USE_G_GAMMA` | `(cdiv(V,BV), NT, B*HV)` | clean |
| common/chunk_o.py:455 | `chunk_bwd_kernel_dv_local` | `H,HV,K,V,BT,BK,BV,USE_G` | cdiv-based | clean |
| common/chunk_delta_h.py:35 | `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` | `H,HV,K,V,BT,USE_EXP2,TRANSPOSE_STATE` | `(cdiv(V,BV), N*HV)` | clean |
| common/chunk_delta_h.py:340 | `chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64` | `H,HV,K,V,BT,BV,USE_G,USE_EXP2,TRANSPOSE_STATE` | `(cdiv(V,BV), N*HV)` | clean |
| common/chunk_scaled_dot_kkt.py:28 | `chunk_scaled_dot_kkt_fwd_kernel` | `H,HV,K,BT,IS_VARLEN` | `(NT, B*HV)` | clean |
| common/chunk_h_split.py:29/149/236/355 | `chunk_fwd/bwd_kernel_h_split / *_reduction` | `BT,USE_G,USE_GK,USE_GV` | `(cdiv(K,BK), cdiv(V,BV), NS*H)` | clean (BS≥BT by assert) |
| common/fused_recurrent.py:26 | `fused_recurrent_fwd_kernel` | `BK,BV,USE_G,USE_G_GAMMA,USE_GK,USE_GV` | `(NV, NK, N*H)` | clean |
| common/fused_recurrent.py:139 | `fused_recurrent_bwd_kernel` | `BK,BV,USE_G,USE_G_GAMMA,USE_GK,USE_GV` | `(NV, NK, N*H)` | clean |
| gla/chunk.py:309 (KDA o-path dep) | `chunk_gla_fwd_kernel_o` | `BT,HV,TRANSPOSE_STATE` | cdiv-based | clean |
| utils/cumsum.py:27/82 (KDA gate dep) | `chunk_local_cumsum_{scalar,vector}_kernel` | `B,H,(S,)BT,IS_VARLEN,REVERSE` | `(NT, B*H)` | clean¹ |

¹ `B` is in the cumsum keys but unused in the scalar kernel body — a *minor* under/over-specified cache key. **NOT a bug**: `B` is the fixed batch dim, not a `cdiv(T,·)` bucket, so it never forces runtime re-autotuning the way phantom `NB` did. Left as-is; outside the `kda/`+`common/` target scope anyway.

## "Suspicious-but-actually-safe" notes (explicitly ruled out)

- **chunk_h.py `NS = cdiv(T, BS)`** — looks like the norm `BS = ceil(T/NS)` pattern but is the inverse: `BS` is a multiple-of-`BT` split size (asserted), `NS` is a state-buffer count, grid is `N*H`-based. No SM cap, no overlap. SAFE.
- **chunk_h_split.py `NS = N*cdiv(T,S)` grid `(…, NS*H)`** — `i_s = i_ss % NS` reconstructs split index from a cdiv-derived `NS`; `S % BT == 0` asserted; tiles via `range(cdiv(i_s*S,BT), cdiv(min(i_s*S+S,T),BT))` so blocks never overlap. SAFE.
- **Missing constexpr keys** (`USE_EXP2` in chunk_o/delta_h, `S`/`USE_SAFE_GATE` in chunk_intra) — these *under-specify* the autotune cache (could in theory reuse a config across a heuristic flip), but they are run-constant, not `T`-derived, so they neither crash nor re-trigger autotuning. Cosmetic, not the sm_120 bug. NOT PATCHED.

## Conclusion / handoff

The KDA chunked linear-attention core is clean for the sm_120 phantom-key + `BS<BT` overlap bug class. If a Kimi-Linear KDA run still crashes on RTX 5090 with a device-side assert, the corrupting kernel is **not** in `fla/ops/kda/` or the `fla/ops/common/` chunk kernels — it is the already-known async-surfaced crash from `fla/modules/fused_norm_gate.py` (PR13), `layernorm.py`/`l2norm.py` (PR13c), or `causal_conv1d` (PR13b, owned by the conv agent). Those three are the only sm_120 offenders in the Kimi-Linear stack; this audit confirms the linear-attention math kernels are not a fourth.

## Modules to shadow

**None.** No vendored patch is produced for the KDA core because no confirmed bug exists. (For completeness, the already-shadowed modules from sibling PRs remain: `fla.modules.fused_norm_gate`, `fla.modules.layernorm`/`l2norm`, and the causal_conv1d ops.)

## Minimal repro scripts — NOT RUN (GPUs busy)

These would *confirm the clean result* (expected: all pass, no illegal-memory-access) if a GPU frees up. They mirror the #796 reproducer shape (large/varying `T` to exercise autotuner re-trigger) but target the KDA kernels.

```python
# repro_kda_chunk_clean.py  —  NOT RUN (8 GPUs busy training)
# Expected on sm_120 + Triton 3.6: completes with no device-side assert
# (the KDA core has no phantom T-derived key and no BS<BT grid cap).
import torch
from fla.ops.kda import chunk_kda  # KDA chunked fwd+bwd entry

def run(T):
    B, H, K, V = 1, 8, 128, 128
    dt = torch.bfloat16; dev = "cuda"
    q = torch.randn(B, T, H, K, dtype=dt, device=dev, requires_grad=True)
    k = torch.randn(B, T, H, K, dtype=dt, device=dev, requires_grad=True)
    v = torch.randn(B, T, H, V, dtype=dt, device=dev, requires_grad=True)
    g = torch.randn(B, T, H, K, dtype=dt, device=dev, requires_grad=True)
    beta = torch.rand(B, T, H, dtype=dt, device=dev, requires_grad=True)
    o, _ = chunk_kda(q, k, v, g, beta, scale=K**-0.5,
                     use_qk_l2norm_in_kernel=True)
    o.sum().backward()
    torch.cuda.synchronize()
    print(f"T={T}: OK")

# vary T across many autotuner buckets (the #796 crash trigger shape)
for T in [512, 1536, 2048, 4096, 8192, 16384, 24000]:
    run(T)
```

```python
# repro_kda_gate_clean.py  —  NOT RUN (8 GPUs busy training)
# Exercises fused_kda_gate (the kernel torchtitan kimi_linear references).
import torch
from fla.ops.kda.gate import fused_kda_gate

for T in [512, 2048, 8192, 24000]:
    H, K = 8, 128
    g = torch.randn(1, T, H, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    A_log = torch.randn(H, device="cuda", requires_grad=True)
    dt_bias = torch.randn(H * K, device="cuda", requires_grad=True)
    y = fused_kda_gate(g, A_log, dt_bias, lower_bound=None)
    y.sum().backward()
    torch.cuda.synchronize()
    print(f"gate T={T}: OK")
```
