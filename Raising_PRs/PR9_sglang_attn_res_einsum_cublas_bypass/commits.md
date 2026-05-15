# Backing commits — PR #9 AttnRes einsum cuBLAS bypass

## Discovered in

**Phase 11** — fp8 weight-only inference sweep on RTX 5090 for the
AttnRes-Kimi-Linear ckpt. The fp8 dense-Linear path enabled,
crashed at the first AttnRes block-aggregation einsum with
`CUBLAS_STATUS_EXECUTION_FAILED on cublasGemmStridedBatchedEx
CUDA_R_16BF`. bf16 path ran fine. `.contiguous()` defense didn't fix
it; manual broadcast+sum sidesteps the GEMM entirely.

## Fork source

| Field | Value |
|---|---|
| Repo | `git@github.com:QIU023/sglang.git` |
| Branch | `attention_residual_inference` (also reachable from `main` after merge `dc154e785`) |
| Commit | `a6c46168a53a57175e3389b11d2067abc3a657b2` (**BUNDLED — needs isolation**) |
| Relevant file | `python/sglang/srt/layers/attn_res.py` |

Bundled with PR #7 (KDA fp16) and PR #8 (MoE Blackwell shmem) — same
commit `a6c46168a`.

## Status

- **Overlay-side fix**: ready, verified bf16 44.6 → 44.6 tok/s
  unchanged + fp8 unblocked at 38.9 tok/s.
- **cuBLAS root cause**: not patched; file a separate reproducer
  issue for NVIDIA / cuBLAS team.

## Filing dependency

Requires **PR #5** (the AttnRes overlay) to land first — the file
`layers/attn_res.py` doesn't exist upstream until then.

## Isolation recipe (after PR #5 lands)

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git remote add qiu023 https://github.com/QIU023/sglang.git
git fetch qiu023
git checkout -b sglang-attn-res-einsum-bypass upstream/main

# Cherry-pick only the attn_res.py file from the bundle.
git checkout qiu023/attention_residual_inference -- \
    python/sglang/srt/layers/attn_res.py

git add python/sglang/srt/layers/attn_res.py
git commit -m "[layers/attn_res] Replace block-aggregation einsums with manual broadcast+sum

Sidestep CUBLAS_STATUS_EXECUTION_FAILED on cublasGemmStridedBatchedEx
CUDA_R_16BF when an upstream tensor came out of an fp8 dequant path
(RTX 5090, SM 12.0). Pure bf16 path works; bf16-after-fp8-dequant
fails. Manual broadcast+sum bypasses the strided GEMM entirely.

bf16 verification: 44.6 tok/s -> 44.6 tok/s (identical FLOPs).
fp8 weight-only verification: from crashed to coherent 8/8 at 38.9
tok/s.

See <cuBLAS reproducer issue link> for the deeper driver-side
investigation. This patch is the overlay-side workaround; the cuBLAS-
side root cause is independent."

git push origin sglang-attn-res-einsum-bypass
```

## Folding option (recommended)

Alternative to filing PR #9 as a follow-up to PR #5: **fold the
broadcast+sum change directly into PR #5's initial commit**. The
overlay ships fp8-compatible from day 1, single PR for reviewers.

Trade-off: PR #5 is already big; adding 3 broadcast+sum lines is
trivial; fold is the cleaner choice.

## Cross-link: cuBLAS reproducer issue (file separately)

In addition to the overlay-side PR, file a cuBLAS reproducer issue
either at NVIDIA (cuda-developer-tools or cuda issues) or as a
cross-link from the SGLang issue tracker:

- Driver version, cuBLAS version, GPU SM cap.
- Minimal repro: produce a bf16 tensor from fp8 dequant, call
  `torch.einsum("n..., n...d -> ...d", w, V)` with N=8, D=1024.
- Confirm pure-bf16-input version works; only fp8-dequant-source
  fails.

This part has no patch; just a reproducer for the maintainers.

## Notes for the PR opener

- The PR description should be **honest about the workaround
  framing**: this is "bypass cuBLAS" not "fix cuBLAS". Otherwise
  reviewers may ask "why not file the cuBLAS bug instead?" —
  answer: both, but the overlay-side workaround unblocks users
  today.
- Verify on H100 too: bf16 throughput unchanged (no GEMM
  acceleration loss on a card where the original einsum works
  fine).
