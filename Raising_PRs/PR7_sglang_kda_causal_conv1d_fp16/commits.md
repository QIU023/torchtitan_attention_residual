# Backing commits — PR #7 KDA `causal_conv1d_triton` fp16 type-join

## Discovered in

**Phase 11** — fp8 / fp16 / bf16 dtype sweep on `hf_step3100` ckpt for
the inference benchmarking matrix (`phase11_rlhf_grpo_infra/bench_inference_dtype.py`).
The fp16 row of the matrix failed at SGLang Engine boot with a Triton
compilation error inside `_causal_conv1d_fwd_kernel`; bf16 and fp32
rows worked.

## Fork source

| Field | Value |
|---|---|
| Repo | `git@github.com:QIU023/sglang.git` |
| Branch | `attention_residual_inference` (also reachable from `main` after merge `dc154e785`) |
| Commit | `a6c46168a53a57175e3389b11d2067abc3a657b2` (**BUNDLED — needs isolation**) |
| Author / date | QIU023 — 2026-05-15 |
| Title | `[Blackwell+fp16/fp8] KDA fp16 type-join + MoE Blackwell shmem + AttnRes cuBLAS bypass` |
| Files in bundle | `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`<br>`python/sglang/srt/layers/attn_res.py`<br>`python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py` |

Status: **bundled with two unrelated fixes; the fp16 patch must be isolated before filing.**

## Why this needs isolation

Commit `a6c46168a` carries three independent fixes for unrelated upstream
bugs, each of which we're filing as a separate PR:

| File | PR | Filing target |
|---|---|---|
| `mamba/causal_conv1d_triton.py` | **PR #7** (this folder) | `sgl-project/sglang` |
| `moe_runner/.../fused_moe_triton_kernels.py` | PR #8 (separate folder, not yet drafted) | `sgl-project/sglang` |
| `layers/attn_res.py` (einsum → broadcast+sum) | PR #9 (separate folder, not yet drafted) | `sgl-project/sglang` |

Filing one combined PR would invite reviewer pushback ("split this into
three"), so we split first.

## Isolation recipe

```bash
# 1. Clone upstream sglang; branch off main.
git clone https://github.com/sgl-project/sglang.git
cd sglang
git remote add qiu023 https://github.com/QIU023/sglang.git
git fetch qiu023

# 2. Branch from latest upstream main.
git checkout -b sglang-kda-causal-conv1d-fp16 upstream/main

# 3. Cherry-pick just the relevant file from the bundle.
git checkout qiu023/attention_residual_inference -- \
    python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py

# 4. Stage + commit with a single-purpose message.
git add python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py
git commit -m "[mamba/causal_conv1d] Fix fp16 dtype type-join in _causal_conv1d_fwd_kernel

Reuse load buffer dtype across the HAS_INITIAL_STATES if/else join so
the Triton compiler can merge both branches under user-requested fp16.
Previously the kernel only accepted bf16 / fp32 because the
HAS_INITIAL_STATES branch read from a bf16 residual buffer while the
opposite branch carried fp16, causing a type-join error.

Verified on Kimi-Linear hf_step3100: fp16 path 44.5 tok/s coherent 8/8;
bf16 baseline 44.6 tok/s regression-clean."

# 5. Add the regression test from PR.md "Test plan".
#    Path: python/sglang/test/srt/layers/attention/test_causal_conv1d_dtype.py
git add python/sglang/test/srt/layers/attention/test_causal_conv1d_dtype.py
git commit -m "[mamba/causal_conv1d] Add fp16/bf16/fp32 dtype regression test"

# 6. Push + open PR.
git push origin sglang-kda-causal-conv1d-fp16
```

## Fork-verification numbers (reuse in PR description)

Verified on `QIU023/sglang@attention_residual_inference` commit
`a6c46168a` against the `hf_step3100` ckpt:

| dtype | tok/s | coherent / 8 prompts |
|---|---|---|
| bf16 baseline | 44.6 | 8 / 8 |
| **fp16 (this PR)** | **44.5** | **8 / 8** |
| fp32 | (slower, unchanged) | 8 / 8 |

No regression on bf16; fp16 throughput matches bf16 within noise.

## Conflict surface

The `_causal_conv1d_fwd_kernel` in
`python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py` is
upstream's verbatim port from the original Mamba reference; rarely
touched. Cherry-pick should apply clean unless upstream has rewritten
the kernel structurally.

If a structural change has happened, hand-port: the fix is "ensure both
sides of the `HAS_INITIAL_STATES` if/else branch have the same Triton
dtype before the join" (see PR.md "Fix sketch" section for two
equivalent patch shapes).

## Notes for the PR opener

- Test file path follows sglang's existing test layout
  (`python/sglang/test/srt/layers/attention/`). Verify the actual
  location matches when filing.
- Pure kernel patch + test, no API change. Should be one of the easier
  PRs to land — no design discussion expected.
