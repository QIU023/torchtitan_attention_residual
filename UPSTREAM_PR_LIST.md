# Upstream PR Candidates

Inventory of changes in this fork that are scope-bounded enough to upstream.
Ordered by **effort × value** (smallest, most-likely-to-land first).

Last updated: 2026-05-10. Status reflects the state as of the overnight
VLM pretrain/SFT/GRPO pipeline run.

| # | Target repo | Title | Scope | Effort | Risk | Status |
|---|---|---|---|---|---|---|
| 1 | sglang | `SGLANG_DISABLE_SHM_MM` env to force CPU mm transport | 9-line patch | XS | low | Ready |
| 2 | sglang | base64 data-URL support in `attn_res_vl` image loader | 6-line patch | XS | low | Ready (depends on #5) |
| 3 | sglang | flashinfer_mla bf16 NaN — repro + fp32 fallback knob | issue + ~150-line patch | M | medium | Ready (issue), patch needs scoping discussion |
| 4 | torchtitan | `parallelize_fn` signature stability for `experiments.rl.PolicyTrainer` | 1-line + adapter trait | S | low | ⛔ **OBSOLETED 2026-05-17** — upstream `627f4a31 [rl] Trainer refactor` (2026-04-20) already widened the kwargs. Do not file. |
| 5 | sglang | Block AttnRes inference overlay (Kimi + Qwen3 carriers) | full new model class + `layers/attn_res.py` | L | high | Research-track |
| 6 | sglang | RS+merge+AG seq-shard fusion documented as a feature | docs + model-hook examples | M | low | Documented in our overlay; needs upstream deciding generality |
| 7 | sglang | KDA `causal_conv1d_triton` fp16 dtype type-join fix | 1-kernel patch + regression test | XS | low | Ready (verified in fork commit `a6c46168a`) |
| 8 | sglang | fp8 weight-only MoE fused kernel Blackwell shmem autotune | autotune row + downstream ICA followup | S | medium | Partial (shmem-shrink in `a6c46168a`); downstream ICA needs deeper Triton debug |
| 9 | sglang | AttnRes block-aggregation einsum bypass for fp8 dequant cuBLAS failure | 3-call manual broadcast+sum | XS | low | Ready (fork commit `a6c46168a`); cuBLAS root cause separate issue |
| 10 | sglang | `Fp8Config.get_quant_method` user-visible warning when MoE silently falls back to bf16 | ~10-line logging | XS | low | Tentative (gated on #8 landing) |
| 11 | pytorch/torchstore | sync-endpoint dispatch policy — allow async caller via flag / endpoint declaration | ~30-line patch + endpoint API | S | medium | Workaround live (Controller monkeypatch in `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py`); upstream form needs API design |
| 12 | torchtitan | engine-agnostic `Generator` abstraction in `experiments.rl` + SGLang reference impl | new module (~600 lines) + RFC | L | medium | Code ready (fork's `experiments/rl/{actors/sglang_generator.py,plugin.py,models/sglang_wrapper.py}` + `RFC_SGLANG_GENERATOR.md`); needs upstream design discussion. Depends on #4 landing first. |

---

## #1 — SGLANG_DISABLE_SHM_MM env

**Target**: `sgl-project/sglang` :: `python/sglang/srt/managers/tokenizer_manager.py`

**Patch** (already in our fork: commit `74083ffae`):

```python
def _determine_tensor_transport_mode(server_args) -> TensorTransportMode:
    if os.environ.get("SGLANG_DISABLE_SHM_MM", "0") == "1":
        return "default"  # inline pickle, lifecycle-safe
    if server_args.dist_init_addr:
        return "default"
    return "cuda_ipc"
```

**Why upstream**: SGLang's POSIX-SHM bridge for multimodal payloads races against
container-actor lifecycles (Monarch / Ray / SLURM job arrays where the spawning
actor unlinks `/psm_*` before the scheduler subprocess opens it).
`FileNotFoundError: '/psm_xxx'` is the symptom. Cross-node already takes the
"default" branch — we just expose the same opt-in for single-node use.

**Why low risk**: env-gated, default behavior unchanged.

**Effort**: 30 min including writing a one-paragraph docstring + adding to docs.

---

## #2 — base64 data-URL in `attn_res_vl` image loader

**Target**: `sgl-project/sglang` :: `python/sglang/srt/multimodal/processors/attn_res_vl.py`
*(this file exists only in our fork; it ships with the AttnRes carrier #5)*

**Patch** (already in our fork: commit `850ebb715`):

```python
if isinstance(item, str):
    if item.startswith("data:image/") and ";base64," in item:
        _, _, payload = item.partition(",")
        return Image.open(BytesIO(b64.b64decode(payload))).convert("RGB")
    return Image.open(item).convert("RGB")
```

**Why upstream**: matches the OpenAI-vision API spec for image inputs (data URLs
are the standard inline format). Useful for RL rollouts and any caller that
wants to avoid disk-side mmap during async data loading.

**Status**: blocks on #5 landing first.

---

## #3 — flashinfer_mla bf16 NaN on high-magnitude residuals

**Target**: `sgl-project/sglang` (issue) + `flashinfer-ai/flashinfer` (fix)

**Repro** (in our fork: `phase11_rlhf_grpo_infra/VISION_INJECTION_BUG_RCA.md` +
`phase11_rlhf_grpo_infra/SGLANG_PR_PROPOSALS.md`):

```
Model: Kimi Linear AttnRes (1.4B-active)
Hardware: RTX 5090 SM 12.0 (Blackwell consumer)
Symptom: NaN logits at deepest MLA layer when prefill input max≈77
Same model + weights via torch eager: works
Other backends on Blackwell: triton OOMs, fa3 needs SM 80-90,
                              torch_native doesn't support MLA layout
```

**Our workaround** (in our fork: commit `e8e7134ee`):
fp32 eager SDPA fallback for MLA layers under EXTEND/prefill, native
flashinfer_mla in DECODE (per-step input is bounded). Cache layout is
preserved so prefill→decode handoff is correct.

**Suggested upstream form**:
- File the bug as a tracked SGLang issue with the minimal repro
- Propose a `--mla-fp32-scoring` flag that runs Q@K + softmax in fp32,
  V multiply in bf16 — or accept a per-layer eager fallback hook that
  third-party model classes can install

**Why this isn't trivially landable**: flashinfer team's call on whether to
expand the kernel API. The per-layer hook approach is more invasive but more
general.

---

## #4 — `parallelize_fn` signature stability for `experiments.rl.PolicyTrainer` (⛔ OBSOLETED 2026-05-17)

> **OBSOLETED-BY-UPSTREAM 2026-05-17.** Upstream commit `627f4a31 [rl]
> Trainer refactor` (2026-04-20) already widened the
> `parallelize_fn` kwargs surface. Do not file this PR. Fork
> reconciliation (delete launcher-side `parallelize_fn` adapter
> wrappers to avoid double-kwarg injection) tracked in
> `Raising_PRs/FORK_REBASE_TASK.md`. PR #12 (engine-agnostic
> Generator) is now unblocked since this prerequisite is already
> upstream.

**Target**: `pytorch/torchtitan` :: `torchtitan/experiments/rl/actors/trainer.py`

**Problem** (in our fork: `phase11_rlhf_grpo_infra/rlhf/run_grpo_kimi_attn_res.py` workaround):

`PolicyTrainer._build_model` calls `model_spec.parallelize_fn(model,
parallel_dims=, parallelism=, compile_config=)` — but core
`parallelize_*` functions (e.g. `parallelize_kimi_linear`,
`parallelize_deepseek_v3`) require four more kwargs: `training`,
`model_converters`, `ac_config`, `dump_folder`. RL trainer can't drive any
non-Qwen3 model_spec without a per-model adapter wrapper.

**Suggested fix** (1-line trainer change + base):
```python
# In trainer._build_model:
parallelize_kwargs = dict(
    parallel_dims=self.parallel_dims,
    parallelism=config.parallelism,
    compile_config=config.compile,
    training=config.training,            # NEW
    model_converters=config.model_converters or default,  # NEW
    ac_config=config.activation_checkpoint, # NEW
    dump_folder=config.dump_folder,      # NEW
)
model = model_spec.parallelize_fn(model, **parallelize_kwargs)
```

**Why upstream**: RL trainer becomes model-agnostic. Removes the need for
per-flavor adapter shims that every non-Qwen3 RL entry-point currently
duplicates.

**Risk**: low — additive kwargs with sensible defaults from existing config.

---

## #5 — Block AttnRes inference overlay

**Target**: `sgl-project/sglang` :: new `python/sglang/srt/layers/attn_res/`
+ `python/sglang/srt/models/{kimi_linear,qwen3}_attn_res_overlay.py`

**Scope** (in our fork: `sglang/python/sglang/srt/{layers/attn_res,models/attn_res_overlay,models/qwen3_attn_res_overlay,multimodal/processors/attn_res_vl,models/attn_res_vl_overlay}.py`):

Block Attention Residual is a generic residual-stream overlay (Kimi paper §5,
ByteDance Hyper-Connections, DeepSeek mHC family). Replaces the standard
pre-norm residual with a learned aggregation over committed prior blocks +
current partial block. Two carriers proven (Kimi Linear MoE, Qwen3 dense)
with 7 vs 1 patches respectively — model-agnostic core extracted.

**What lands**:
- `layers/attn_res.py` — algorithm only (block_attn_res, two-phase batched
  variant, RS+merge+AG seq-shard fusion, zero-init pseudo query, all-gather
  helpers). Not model-specific.
- Per-model overlay = thin wrapper exposing a `EntryClass` that SGLang's
  registry picks up via `architectures: ["XxxBlockAttnResForCausalLM"]`.
- Optional VL counterpart for VLM carriers.

**Why this is research-track not just a feature**:
- Requires the paper / academic legitimization to land (otherwise looks
  like a one-off weird-residual variant)
- Touches model-loader assumptions (dual-arch hint, fp32 MoE bias patch,
  RMSNorm contiguous shim) — all small but not zero coupling

**Status**: code exists, runs, two carriers validated, NCCL traces archived.
PR would be an RFC + staged landing (algorithm-first, then carriers).

---

## #6 — RS+merge+AG seq-shard fusion documented as a feature

**Target**: `sgl-project/sglang` :: docs + a `model_runner_hooks.py` example

**Background**: In our overlay we replace the default per-layer
`AllReduce(o_proj)` with `ReduceScatter(o_proj_partial) + cross-layer
merge + AllGather`, which halves the on-wire AllReduce volume across a
PP/TP group. The fusion is generic — applies to any TP=N model with
`o_proj.reduce_results=False` + a cooperative scheduler that knows which
layers can stage their AllReduce together.

**Why upstream**: documented separately because the algorithm is small but
the *plumbing* (model-runner hook to defer reduces, cross-layer barrier
groups) is generic and would benefit other TP-heavy models.

**Status**: only documented in our overlay's comments and the Phase 11
trace catalog. Needs lifting into a standalone proposal.

---

## #7 — KDA `causal_conv1d_triton` rejects fp16 dtype

**Target**: `sgl-project/sglang` :: `python/sglang/srt/layers/attention/mamba/causal_conv1d_triton.py`

**Status**: implemented in fork (`qiu023/sglang` `attention_residual_inference` commit `a6c46168a`). Smoke-verified on hf_step3100: fp16 path 44.5 tok/s coherent 8/8 (vs bf16 baseline 44.6). bf16 regression-clean.

**Symptom**: booting an SGLang Engine with `dtype="float16"` for any
KDA-using model (Kimi-Linear and friends) crashes at first KDA layer
with:

```
triton.compiler.errors.CompilationError: at 105:8:
AssertionError("Mismatched type for col0 between then block
                (<['256'], bf16>) and else block (<['256'], fp16>)")
```

The `_causal_conv1d_fwd_kernel` has a branch (`if HAS_INITIAL_STATES:
... if load_init_state: ...`) where one side loads from a buffer that
got promoted to bf16 while the other carries the user's fp16 model
dtype — the type-merge for the `tl.if/else` join fails.

**Why upstream**: blocks fp16 inference for the whole Kimi-Linear /
hybrid-linear-attention model family. Doesn't matter at training
(everyone trains in bf16) but appears whenever someone tries fp16
inference for memory/throughput.

**Fix sketch**: cast both branches to a common dtype before the join,
or use `tl.where` instead of `if/else` for the single-tensor select.

**Effort**: 1 hour (one kernel + one regression test).

---

## #8 — fp8 weight-only MoE fused kernel exceeds RTX 5090 shared memory

**Target**: `sgl-project/sglang` ::
`python/sglang/srt/layers/moe/moe_runner/triton_utils/fused_moe_triton_kernels.py`

**Status**: **partial** — implemented in fork (`qiu023/sglang` `attention_residual_inference` commit `a6c46168a`). The shmem-shrink helper `_maybe_shrink_config_for_sm120` does fix the OutOfResources crash (verified: kernel launches now succeed). **However**, the shrunk config (BLOCK_M=64, num_stages=2, num_warps=4) still triggers a downstream "Triton Error [CUDA]: an illegal memory access" inside the fp8 fused-MoE kernel on RTX 5090 — likely a separate issue with the kernel's pipelining or expert-token-counting under reduced num_stages on SM 12.0 (needs deeper Triton-level debugging than this PR scope). Smoke workaround: skip MoE quant entirely via `SGLANG_FP8_IGNORED_LAYERS=...,mlp.experts` so MoE stays bf16; fp8 weight-only on dense Linear (q/k/v/o, mlp gate/up/down) still works (38.9 tok/s coherent 8/8 vs bf16 baseline 44.6). Filing this PR makes that workaround unnecessary for users who only need fp8 dense-Linear, AND lays groundwork for a follow-up PR that finishes the Blackwell MoE path.

**Symptom**: `Engine(dtype="bfloat16", quantization="fp8", ...)` for a
Kimi-Linear MoE model on Blackwell (RTX 5090, SM 12.0) crashes at
first MoE forward with:

```
triton.runtime.errors.OutOfResources: out of resource: shared memory,
Required: 147456, Hardware limit: 101376
```

The fp8 fused-MoE kernel was tuned for SM 9.0+ (Hopper/H100, ≥228KB
shared memory). RTX 5090 (Blackwell consumer, 100KB shared memory) is
underestimated — block sizes and `num_stages` need a smaller-shmem
tuning autoconfig path.

**Why upstream**: Blackwell consumer cards (RTX 5090 / 5080) are
becoming common rental hardware, and fp8 weight-only quantization is a
key throughput win. Without this, Kimi-Linear-class MoE models can't be
fp8-served on these cards at all — the fall-through to bf16 weights
defeats the quantization point.

**Fix sketch**: add a `BLOCK_SIZE_M=64` (or smaller) + `num_stages=2`
(or 1) tuning row to `_get_default_config` for Blackwell consumer
target, gated by `device_capability == (12, 0)` and
`shared_memory_per_block < 128KB`.

**Effort**: ~2 hours including autotune sweep.

---

## #9 — AttnRes block-aggregation einsums crash cuBLAS strided batched bf16 GEMM under fp8 paths

**Target**: `sgl-project/sglang` :: `python/sglang/srt/layers/attn_res.py` (in our overlay) — but the upstream-relevant story is the cuBLAS GEMM behaviour itself, see "Why upstream" below.

**Status**: implemented in fork (`qiu023/sglang` `attention_residual_inference` commit `a6c46168a`). Smoke-verified: fp8 weight-only path now coherent 8/8 on hf_step3100 (38.9 tok/s) vs prior CUBLAS_STATUS_EXECUTION_FAILED.

**Patch**: replace three einsums of the form `("[q]n..., n...d -> [q]...d")` (small N=8 contraction → large D=1024 output) in `block_attn_res()`, vectorised `block_attn_res_phase1()`, and the per-query fallback, with manual `(weights.unsqueeze(-1) * V).sum(dim=0)`. Bypasses cuBLAS's strided batched GEMM entirely (the natural einsum decomposition for this shape lands there). bf16 path: identical FLOPs, no measurable throughput change (44.6 → 44.6 tok/s); fp8 path: now works.

**Why this is partly an upstream concern**: the cuBLAS error itself is `CUBLAS_STATUS_EXECUTION_FAILED on cublasGemmStridedBatchedEx CUDA_R_16BF` only when an upstream tensor came out of an fp8 dequant path — same shapes/strides, same dtype, same kernel — works fine in pure bf16, fails after fp8. Either:
* an alignment requirement of the cuBLAS Blackwell bf16 GEMM kernel that fp8 dequant violates (would benefit cuBLAS / driver-side handling), or
* a stride/storage assumption SGLang's fp8 dequant path produces that cuBLAS rejects (would benefit a `.contiguous()` insertion in the dequant return path).

The overlay-side `.contiguous()` defense was tried first — does NOT fix it (rules out simple contiguity). The manual-broadcast fix sidesteps the GEMM but doesn't explain *why* cuBLAS rejects this specific combination. A reproducer is straightforward: any model with an fp8-quantized layer feeding into `torch.einsum("n..., n...d -> ...d", w, V)` with N≤16 and D≥512 on RTX 5090.

**Effort**: 1 hour for the overlay-side fix (already done, committed in fork). Untangling the cuBLAS-side root cause is a separate investigation — may belong on the cuBLAS/driver side, not SGLang.

---

## #10 — `Fp8MoEMethod` does not pass `ignored_layers` filter cleanly for partial-MoE skip

(Tentative — needs more investigation before filing.)

**Symptom**: `Fp8Config.get_quant_method` does honor `ignored_layers` for both `LinearBase` and `FusedMoE` (returns `UnquantizedFusedMoEMethod`), and the env knob `SGLANG_FP8_IGNORED_LAYERS=...,mlp.experts` is the documented escape hatch. Works correctly. **However**, if PR #8 lands a "best-effort" Blackwell MoE that still ICAs (the open-issue note in #8), the only practical config for fp8 weight-only on Blackwell becomes "mlp.experts in the ignored list" — which is fine but silent: there's no warning that the user is paying for fp8 quant on dense Linear *only* and getting bf16 MoE under the hood.

**Proposal**: when `Fp8Config.get_quant_method` returns `UnquantizedFusedMoEMethod` for a `FusedMoE` layer due to `ignored_layers`, log a one-line INFO so users understand their effective quant scheme. ~10 lines of logging.

**Status**: not implemented; flagged for future filing if/when PR #8's downstream ICA is resolved (otherwise this is the only viable fp8 path on Blackwell consumer cards and warrants user-facing visibility).

---

## #11 — torchstore sync-endpoint dispatch policy (async caller opt-in)

**Target**: `pytorch/torchstore` :: endpoint dispatch policy.

**Symptom**: torchstore's `Controller` rejects an async caller hitting a
sync endpoint (and vice versa). Concrete blocker: in our GRPO RL loop
(`phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py`), the actor mesh's `Generator`
is async (SGLang Engine returns a coroutine), but torchstore's 5
endpoints used for weight-sync and state-broadcast (`put`, `get`,
`broadcast`, `barrier`, `shutdown`) are declared sync. Result: hard
exception at the first cross-mesh call. The runtime API has no
documented escape hatch.

**Our workaround** (in fork: `phase11_rlhf_grpo_infra/rlhf/run_grpo_llava_kimi.py`):
monkey-patch the `Controller` at process start in BOTH the main
process AND every Monarch-spawned subprocess to wrap the 5 sync
endpoints into thin async-coroutine adapters. **0 performance impact**
(the adapters are passthrough), but the patching machinery is fragile —
adding a new endpoint upstream silently breaks our wrapper.

**Same pattern as PR #1**: upstream-side strict policy, add an opt-in
flag for callers that know what they're doing. Two viable upstream
shapes:

1. **Endpoint-level declaration** (preferred): let endpoint authors
   declare `dispatch_mode={sync, async, auto}`. `auto` accepts either
   and wraps internally. Backwards-compatible (default `sync` keeps
   today's behavior).

2. **Env-gated bulk override**: `TORCHSTORE_ALLOW_MIXED_SYNC_ASYNC=1`
   relaxes the dispatch policy globally. Lighter to land, less clean
   long-term.

**Suggested upstream form**: file an issue with the GRPO repro
(actor mesh = SGLang async, controller mesh = torchstore sync,
5-endpoint cross-mesh exchange), let torchstore maintainers decide
between (1) and (2) above. Patch follows the chosen direction.

**Why this isn't trivially landable**: API design choice — pure
dispatch-mode-on-endpoint is cleaner but more invasive; env flag
is simpler but lasting (env-config debt). Upstream call.

**Status**: workaround live in fork; issue not yet filed. ~30 line
patch once API direction decided.

---

## #12 — Engine-agnostic `Generator` abstraction in `experiments.rl` + SGLang reference impl

**Target**: `pytorch/torchtitan` :: `torchtitan/experiments/rl/` (new
modules + RFC). **Zero core changes** — entirely within `experiments/`.

**Scope** (in our fork: torchtitan `attention_residual_dev` branch,
9 commits accumulating to ~600 lines + an RFC doc):

- `experiments/rl/actors/sglang_generator.py` — concrete SGLang Engine
  Generator implementing the abstract `Generator` contract (`generate`,
  `update_weights_from_dcp`, `shutdown`).
- `experiments/rl/actors/eager_generator.py` — fallback eager
  PyTorch generation Generator (no SGLang dependency). Useful for
  smoke tests + CI + environments where SGLang isn't installed.
- `experiments/rl/models/sglang_wrapper.py` — SGLang HF-config bridge.
- `experiments/rl/plugin.py` — engine-agnostic plugin registry.
- `experiments/rl/RFC_SGLANG_GENERATOR.md` — design doc.

**Why upstream**: torchtitan's existing `experiments/rl` is tightly
coupled to Qwen3 + a hardcoded generation backend. Engine-agnostic
`Generator` lets RL trainers swap the rollout backend (SGLang for
production, eager for CI, vLLM in the future) without touching
trainer / actor / store code. Same shape as torchtitan's existing
`Tokenizer` / `Optimizer` plugin pattern.

**Why this isn't trivially landable**: new abstraction layer requires
upstream design discussion. Reviewer concerns expected:
- Generator interface granularity (per-call vs streaming vs batched)
- How `update_weights_from_dcp` fits the existing trainer-side
  checkpoint conventions
- Whether SGLang dependency is opt-in (it is in our impl — gated by
  successful `import sglang`)

**Depends on**: PR #4 (parallelize_fn signature stability) needs to
land first, otherwise the Generator can't drive a non-Qwen3
trainer's `_build_model`.

**Suggested upstream form**: file the RFC first (the
`RFC_SGLANG_GENERATOR.md` in our fork is the seed), let torchtitan
team weigh in on interface boundaries before code review.

**Status**: code complete, runs in our overnight GRPO chain. RFC
ready to file standalone. Code PR follows RFC discussion.

---

## What's NOT on this list (and why)

- **fp32 MLA fallback "as a flag"** — too narrow. The flashinfer issue
  (#3) is the real fix. If the issue gets a proper resolution, the fallback
  becomes obsolete.
- **MonarchRPC torchstore transport choice in `run_grpo_*.py`** — workaround
  for a specific container's `ulimit -l = 64KB`. Not a torchstore upstream
  concern; users can configure transport.
- **MoE expert-routing CUDA assert mitigation (retry-with-seed-bump in
  pipeline orchestrator)** — workaround in our launcher, not a fix to
  the underlying kernel. Real fix is upstream `fla-core` /
  `fused_moe_triton` numerical-edge-case hardening — that's a deep
  investigation, not a PR ready to file.
- **`grader_mesh` `sys.path` bootstrap fix** — our own launcher bug, not
  a Monarch upstream concern. Fixed in fork.

## Recommended order

Filing strategy refreshed 2026-05-17 after PR #4 obsoleted-by-upstream:

1. **#1** first (30-min sglang env-gated PR; fork already has the patch
   in `74083ffae`; smallest possible first contribution to build
   reviewer credibility). **Already branched locally** awaiting push.
2. **#7** in parallel with #1 (1-hour sglang kernel + test; fork patch
   file-isolated from `a6c46168a` already branched locally; unblocks
   fp16 inference for the Kimi-Linear family).
3. **#3 issue** without the fix (let flashinfer team weigh in on how they
   want to expose fp32-scoring).
4. **#11 issue** for torchstore (API design discussion).
5. **#8** with the partial shmem-shrink patch (downstream ICA followup
   tracked separately).
6. **#6** as a design RFC (no code change yet).
7. **#5** after the algorithm has a paper or arxiv writeup to point to
   (Kimi K-series release).
8. **#9 issue** with the manual broadcast+sum as a workaround + the
   cuBLAS reproducer pointing toward driver-side investigation. File
   alongside or fold into #5.
9. **#2** after #5 lands (or fold into #5 as a day-1 processor feature).
10. **#12 RFC** for the engine-agnostic Generator abstraction.
    **Unblocked** — the upstream `627f4a31` refactor that obsoleted #4
    satisfies #12's prerequisite. File the RFC anytime; code PR follows
    RFC discussion.
11. **#10** if/when #8's downstream ICA is resolved.

~~**#4** — obsoleted by upstream `627f4a31` on 2026-05-17. Do not file.~~
