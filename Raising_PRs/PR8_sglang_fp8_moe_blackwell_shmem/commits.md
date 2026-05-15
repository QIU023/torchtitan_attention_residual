# Backing commits — PR #8 fp8 MoE Blackwell shmem autotune

## Discovered in

**Phase 11** — fp8 quantization sweep on RTX 5090 for the Kimi-Linear
AttnRes inference benchmark (`phase11/bench_inference_dtype.py`). The
fp8 row crashed at first MoE forward with `OutOfResources: out of
resource: shared memory`. The shmem-shrink helper was added inline
during the sweep to unblock the bench.

## Fork source

| Field | Value |
|---|---|
| Repo | `git@github.com:QIU023/sglang.git` |
| Branch | `attention_residual_inference` (also reachable from `main` after merge `dc154e785`) |
| Commit | `a6c46168a53a57175e3389b11d2067abc3a657b2` (**BUNDLED — needs isolation**) |
| Relevant file | `python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py` |

The shmem-shrink helper `_maybe_shrink_config_for_sm120` is bundled with
PR #7 (KDA fp16) and PR #9 (AttnRes einsum) inside the same commit.

## Status

- **Patch (launch path)**: ready in fork; needs isolation from
  `a6c46168a`.
- **Downstream ICA**: unresolved; file as a separate follow-up issue.

## Isolation recipe

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git remote add qiu023 https://github.com/QIU023/sglang.git
git fetch qiu023
git checkout -b sglang-fp8-moe-blackwell-shmem upstream/main

# Cherry-pick only the MoE file from the bundle.
git checkout qiu023/attention_residual_inference -- \
    python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py

git add python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py
git commit -m "[moe/fp8] Add Blackwell consumer (SM 12.0) autotune row

Add a smaller-shmem autotune row gated by device_capability == (12,0)
and shared_memory_per_block < 128KB, so the fp8 fused-MoE kernel
launches on RTX 5090 / 5080. Existing SM 9.0+ (H100) rows unchanged.

Known limitation: the shrunk config triggers a downstream illegal
memory access on RTX 5090 (filed as <follow-up issue link>). Users
hitting the downstream ICA can set
SGLANG_FP8_IGNORED_LAYERS=...,mlp.experts to keep MoE in bf16 while
fp8 weight-only applies to dense Linear; verified 38.9 tok/s coherent
8/8 vs bf16 baseline 44.6 on hf_step3100."

git push origin sglang-fp8-moe-blackwell-shmem
```

## Filing order

1. File the downstream-ICA issue **first** (before opening the PR) so
   the PR body can link it.
2. Then open the PR with the autotune row + ICA-acknowledgement.
3. After the PR lands, the downstream-ICA issue stays open until
   someone (likely a deeper Triton-MoE maintainer) takes it.

## Conflict surface

The `_get_default_config` function inside
`fused_moe_triton_kernels.py` is occasionally touched by upstream
MoE refactors. Cherry-pick may not apply clean; if it doesn't,
hand-port: just add one `if dev_cap == (12, 0) and shmem < 128KB:`
branch with the BLOCK_M=64 / num_stages=2 / num_warps=4 config.

## Notes for the PR opener

- This is a **partial fix** PR. Be upfront in the body that the
  launch path is fixed but a downstream ICA needs followup.
- Don't oversell: it's "fp8 weight-only on dense Linear works on
  Blackwell" not "fp8 MoE works on Blackwell".
- Numbers from the smoke: 38.9 tok/s fp8-dense + bf16-MoE vs 44.6
  bf16 baseline on hf_step3100. Mention these in the PR description.
