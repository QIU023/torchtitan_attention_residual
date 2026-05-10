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
| 4 | torchtitan | `parallelize_fn` signature stability for `experiments.rl.PolicyTrainer` | 1-line + adapter trait | S | low | Ready |
| 5 | sglang | Block AttnRes inference overlay (Kimi + Qwen3 carriers) | full new model class + `layers/attn_res.py` | L | high | Research-track |
| 6 | sglang | RS+merge+AG seq-shard fusion documented as a feature | docs + model-hook examples | M | low | Documented in our overlay; needs upstream deciding generality |

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

**Repro** (in our fork: `phase11/VISION_INJECTION_BUG_RCA.md` +
`phase11/SGLANG_PR_PROPOSALS.md`):

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

## #4 — `parallelize_fn` signature stability for `experiments.rl.PolicyTrainer`

**Target**: `pytorch/torchtitan` :: `torchtitan/experiments/rl/actors/trainer.py`

**Problem** (in our fork: `phase11/rlhf/run_grpo_kimi_attn_res.py` workaround):

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

## Recommended order

1. **#4** first (1-day torchtitan PR, low risk, removes a pain point for
   anyone trying to RL-train non-Qwen3 models)
2. **#1** in parallel (30-min sglang PR, gated env var)
3. **#3 issue** without the fix (let flashinfer team weigh in on how they
   want to expose fp32-scoring)
4. **#6** as a design RFC (no code change yet)
5. **#5** after the algorithm has a paper or arxiv writeup to point to
6. **#2** after #5 lands
