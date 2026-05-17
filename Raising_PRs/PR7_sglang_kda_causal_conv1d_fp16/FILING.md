# PR #7 — filing instructions

## Status

🟢 **Branch pushed; smoke verified on RTX 4070Ti SM 8.9; PR not yet opened.**

| Item | Link / value |
|---|---|
| Fork branch | https://github.com/QIU023/sglang/tree/pr7-kda-causal-conv1d-fp16 |
| Open-PR URL | https://github.com/QIU023/sglang/pull/new/pr7-kda-causal-conv1d-fp16 |
| Target repo | https://github.com/sgl-project/sglang |
| Base | `sgl-project/sglang:main` |
| Head | `QIU023/sglang:pr7-kda-causal-conv1d-fp16` |
| Commit | `4dfd8cf27` (1 commit, +38/-27 in `causal_conv1d_triton.py`) |
| Verification | 6/7 dtype cases PASS on 4070Ti — see `smoke_kernel_direct_fp16.py` + `smoke_kernel_decode_fp16.py` |
| Cross-link | none required (orthogonal to PR #1 / #8) |

## To open the PR

1. Open https://github.com/QIU023/sglang/pull/new/pr7-kda-causal-conv1d-fp16
2. Confirm base = `sgl-project/sglang:main`, head = `QIU023/sglang:pr7-kda-causal-conv1d-fp16`
3. Title + body below (the latter pulled from [PR.md](PR.md) — already updated with smoke + fp8 sections)

---

## Title (copy-paste)

```
[mamba/causal_conv1d] Fix _causal_conv1d_fwd_kernel fp16 dtype type-join error (KDA / Kimi-Linear fp16 inference)
```

## Body

Use [PR.md](PR.md) verbatim. Key sections it already contains:

- Summary + symptom traceback
- Root cause + fix sketch
- **Direct-kernel smoke matrix** on RTX 4070Ti (added 2026-05-17): prefill 3/4, decode 4/4, KERNEL_WIDTH=4 matching Kimi-Linear `short_conv_kernel_size`
- **fp8 quant interaction**: weight-only fp8 + fp16 activations exercises this patch; verified
- **Follow-up note**: symmetric write-back SSA join in fwd kernel under inverted dtype (not a SGLang real-world config)

## Smoke artifacts to attach

The two smoke scripts in this folder are runnable independently:

```bash
# Both require torch + triton + a CUDA-visible GPU
python Raising_PRs/PR7_sglang_kda_causal_conv1d_fp16/smoke_kernel_direct_fp16.py
python Raising_PRs/PR7_sglang_kda_causal_conv1d_fp16/smoke_kernel_decode_fp16.py
```

Sample output (RTX 4070Ti SM 8.9, torch 2.11.0+cu130, triton 3.6.0):

```
Prefill (_causal_conv1d_fwd_kernel):
  baseline_bf16_bf16             PASS
  bug_repro_fp16_x_bf16_state    PASS  ← PR #7 fixes
  inverted_bf16_x_fp16_state     FAIL  ← follow-up (write-back path; not a SGLang real-world config)
  all_fp16                       PASS

Decode (_causal_conv1d_update_kernel):
  baseline_bf16_bf16             PASS
  bug_repro_fp16_x_bf16_state    PASS  ← PR #7 fixes
  inverted_bf16_x_fp16_state     PASS
  all_fp16                       PASS
```

## Reviewer hints

- The patch is a 1-line semantic change (introduce `col_dtype:
  tl.constexpr = x_ptr.dtype.element_ty`) plus mechanical `.to(col_dtype)`
  casts on the four loads from `conv_states`. Same fix mirrored in
  `_causal_conv1d_update_kernel`.
- `bf16+bf16` default path is byte-identical: `.to(bf16)` of a bf16 load is a
  no-op, and `tl.zeros(..., dtype=bf16)` is exactly what existed before.
- The companion downstream fixes from the original bundle commit
  (`a6c46168a`) — fp8 MoE Blackwell shmem (filed as separate PR #8) and
  AttnRes einsum cuBLAS bypass (deferred per #5 dependency) — are *not*
  bundled here.

## Related work in same batch

- **PR #1** (SHM-MM env-gate) — separate sglang PR, same fork
- **PR #8** (fp8 MoE Blackwell shmem) — separate sglang PR, same fork; same `a6c46168a` bundle
- **PR #9** (AttnRes einsum cuBLAS bypass) — overlay-side workaround, blocked on PR #5
