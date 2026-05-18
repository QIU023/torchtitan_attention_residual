# SGLang Fork — Current State & Production-Ready Gaps

**Date**: 2026-05-18
**Audience**: project lead, future maintainers

This file is the single source of truth for what's in our SGLang fork, how
it's wired today, and what's blocking a clean production-ready release.

---

## 1. Two functionalities, both already merged on AR branch

There used to be confusion that "GRPO works on one branch, low-precision
inference on another". The reality:

```
qiu023/sglang remote
├── main                                — upstream sync (rare use)
├── vlm-sglang-overlay                  — early scratch branch (deprecated)
└── attention_residual_inference (AR)   ← PRIMARY, fully merged
```

The merge commit `dc154e785` (2026-05-15) brought **qiu023/main → AR**:

```
dc154e785  merge qiu023/main: SGLANG_DISABLE_SHM_MM env + fp32 MLA fallback
           (extend-only + write cache) + base64 data-url + NaN trace
           into AR_inference primary branch
```

So **AR is the only branch we use**. Both pieces live there:

| Capability | Commits on AR |
|---|---|
| AttnRes VLM overlay (model + config + layer + image loader) | b3f6b543f, d6fb3bbd7, 850ebb715, c07392916 |
| KDA fp16 type-join (Blackwell) | a6c46168a |
| MoE Blackwell shmem fix | a6c46168a |
| AttnRes cuBLAS bypass (avoid sm_120 GEMM bug) | a6c46168a |
| fp32 MLA fallback (extend-only + write cache, fixes flashinfer_mla NaN) | e8e7134ee, 334990612 |
| SGLANG_DISABLE_SHM_MM env (CPU tensor transport, monarch/torchstore IPC) | 74083ffae |
| base64 data-url image loader | 850ebb715 |
| NaN trace instrumentation | c07392916 |
| causal_conv1d Triton kernel fix | (within a6c46168a series) |

**GRPO rollout + low-precision VLM inference share the SAME branch and
the SAME binary.** There is no merge to do remotely.

## 2. Local working tree state is messy (separate issue)

`pip show sglang` points at `/sgl-workspace/sglang/python` (editable
install). The **working tree there is in detached HEAD** at upstream
`v0.5.11` tag, with 7 cherry-picked files overlaid:

```
HEAD = 612785ffd (vanilla v0.5.11)  ← NOT a branch
working tree edits (vs HEAD; all match qiu023/AR file-for-file, diff=0):
   M python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py
   M python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py
   M python/sglang/srt/managers/tokenizer_manager.py
   M python/sglang/srt/model_executor/model_runner.py
   ?? python/sglang/srt/configs/kimi_attn_res_vl.py
   ?? python/sglang/srt/layers/attn_res.py
   ?? python/sglang/srt/models/attn_res_overlay.py

AR branch is 12,321 commits ahead (~upstream sync + our iterations)
```

**Functional effect today**: GRPO trace and inference work because the 7
patched files cover the critical paths. But:

- We are missing **all upstream sglang fixes post-v0.5.11** (perf, bugs,
  new model adapters, kernel updates)
- We cannot `git pull qiu023 attention_residual_inference` without
  conflicts (HEAD diverged via "cherry-pick onto vanilla")
- Detached HEAD = any local commits become dangling, lost to GC
- New maintainers cannot derive the running state from `git log`

**Reproducibility-blocking**: yes. Fix path documented in §5 below.

## 3. Production-ready gaps (root cause work, NOT workarounds)

Listed by severity. The user explicitly requested root-cause fixes, not
workarounds, so each item names what we currently do and what the real
fix would be.

### P0 — correctness / crash

| # | Issue | Current state | Root-cause fix needed |
|---|---|---|---|
| P0.1 | flashinfer_mla NaN on Blackwell (sm_120) for KDA chunk states | **WORKAROUND**: fp32 MLA fallback (e8e7134ee, 334990612). Slower but avoids NaN. | Real fix: flashinfer kernel bug in `mla_decode` for fp16 with chunk_size > 1 on sm_120. File upstream issue with min-repro; until kernel fix, can't use fast path. **Action**: write min-repro, file `flashinfer-ai/flashinfer` issue. |
| P0.2 | fla KDA `chunk_gated_delta_rule_fwd_h` CUDA device-side assert on sm_120 (training-time, but same kernel used in inference) | **WORKAROUND**: in *training*, dataset shuffle rotates bad samples (today's fix). In *inference*, we haven't hit it yet because rollout sequences are shorter. | Real fix: file `fla-org/flash-linear-attention` issue with min-repro. Likely needs Triton kernel rewrite of `chunk_delta_h.py:695` for sm_120 alignment. **Action**: write min-repro, file upstream issue. PR13 was wrong root cause and was rolled back. |
| P0.3 | SGLANG SHM IPC crash w/ monarch + torchstore | **WORKAROUND**: `SGLANG_DISABLE_SHM_MM=1` env forces CPU tensor transport (74083ffae) | Real fix: SHM segment lifecycle bug when sglang server holds shm fd across monarch actor restarts. Either patch sglang shm cleanup or upstream-fix monarch's actor restart sequence. Adds CPU<->GPU memcpy per request currently — material perf cost. |
| P0.4 | AttnRes cuBLAS GEMM produces wrong output on sm_120 | **WORKAROUND**: cuBLAS bypass for AttnRes projector (a6c46168a) — uses naive matmul instead. | Real fix: this is a cuBLAS / cuBLASLt bug for specific (M,K,N) shapes on sm_120. File NVIDIA bug or wait for cuBLAS update. Currently slower for the AttnRes path. |
| P0.5 | KDA fp16 type-join compile error | **WORKAROUND**: explicit upcast to fp32 mid-kernel (a6c46168a) | Real fix: Triton compiler type inference bug; needs upstream Triton 3.1+ fix or kernel rewrite. |

### P1 — perf / completeness

| # | Issue | Current state | Real fix |
|---|---|---|---|
| P1.1 | MoE Blackwell shared-memory layout reads wrong tile | **PARTIAL FIX** in `fused_moe_triton_kernels.py` (a6c46168a) | Audit other MoE shapes; current fix is shape-specific not general. |
| P1.2 | Detached HEAD on vanilla v0.5.11 — missing 12K upstream commits | hybrid local state | Properly check out AR branch (see §5) |
| P1.3 | `attn_res_overlay.py` model registration is monkey-patched at import | works but not upstream-PR'd | File upstream sglang PR adding `kimi_attn_res_vl` model_type; clean adapter pattern |
| P1.4 | NaN trace instrumentation always on | small perf hit | Wrap in env flag `SGLANG_AR_NAN_TRACE=1`; off by default for prod |
| P1.5 | base64 image loader copy overhead | every image: base64 decode + PIL re-encode | Direct bytes → SigLIP processor without re-encoding through PIL |
| P1.6 | Float8GroupedMMConverter not in dispatch | MoE experts stay bf16 (no fp8 win for experts) | Upstream torchao perf-prototype → integrate when stable |
| P1.7 | causal_conv1d Triton kernel fix is local patch | one-off | File torch / fla upstream issue |

### P2 — release hygiene

| # | Issue | Real fix |
|---|---|---|
| P2.1 | `qiu023/main` and `vlm-sglang-overlay` are legacy branches | Delete or archive after AR is confirmed sole source of truth |
| P2.2 | No CI for our patches | Add minimal pytest under `tests/attn_res/` for VLM overlay, base64 loader, fp32 MLA path |
| P2.3 | No tagged release on our fork | Tag `v0.5.11-kimi-ar-v1` once §5 cleanup done |
| P2.4 | Validation sprint reports (V1-V15) live in `phase11_rlhf_grpo_infra/sglang_validation/` but not summarized in this doc | TODO: link the V0 REPORT.md here once stable |

## 4. What works today (regression baseline)

The validation sprint V1-V15 (task #53-#67) passed on the current
hybrid working tree. Tests covered: bf16/fp16/fp8 logit KL, MoE routing,
fp32 reference parity, TP=2/4/8, long context 4-16K, long generation
2-4K, batch sweep, concurrent load, soak test, edge cases, OpenAI API,
streaming, chat template, logit bias. **No regressions.**

GRPO trace (task #51) runs 60 steps bf16 with NCCL trace + ixia post-
process on Kimi AttnRes (working, infrastructure validated).

**Production-quality runs are blocked by P0.1-P0.5 root-cause fixes
(currently workaround), NOT by validation gaps.**

## 5. Cleanup plan (when GRPO is idle)

```bash
cd /sgl-workspace/sglang

# Verify working tree matches AR exactly (the audit we did 2026-05-18 confirms this)
for f in $(git diff --name-only) $(git ls-files --others --exclude-standard); do
    git diff qiu023/attention_residual_inference -- "$f" | head -1 || echo "DIFFERS: $f"
done

# Move untracked + tracked changes safely
git stash push -u -m "v0.5.11+kimi overlay snapshot 2026-05-18"

# Check out AR; verify it's clean and current
git checkout attention_residual_inference
git pull qiu023 attention_residual_inference
git status   # expect clean

# Stash should now be redundant (files identical); drop it
git stash list
git stash drop  # only after confirming `git diff stash@{0}` is empty

# Optional: tag the validated state
git tag -a v0.5.11-kimi-ar-v1 -m "Validated 2026-05-18 — V1-V15 pass + GRPO trace pass"
git push qiu023 v0.5.11-kimi-ar-v1
```

After this:
- `git log` cleanly shows our work history
- Future upstream pulls via `git fetch upstream && git merge upstream/main` work
- New maintainers can `git checkout v0.5.11-kimi-ar-v1` to reproduce

## 6. Out-of-scope (do not work on until in-scope is done)

- sglang-omni evaluation (whether to switch / mirror omni features) —
  pending sglang-omni repo and changelog review. Don't move until P0 fixes done.
- Float8 grouped GEMM for MoE experts — wait for upstream stability.
- MMMU eval (~25GB) — skip until Priority A/B benchmarks scored.

---

**Owner**: project lead / @QIU023
**Last audit**: 2026-05-18 04:30Z
