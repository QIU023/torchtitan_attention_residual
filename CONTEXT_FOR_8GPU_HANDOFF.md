# Context handoff — AttnRes in torchtitan (8-GPU Phase 3/4)

**This document is for the Claude instance running on the 8-GPU rental
box.** Read in full before taking any action. Your local state (new
machine, new session) starts empty; this rehydrates it.

---

## 1. Project in one paragraph

Implementing **Block Attention Residuals** from *Kimi Team, 2026*
([arXiv 2603.15031](https://arxiv.org/abs/2603.15031)) in
pytorch/torchtitan. Pre-work (Phase 0–2) is done: algorithm primitive
written, single-GPU FSDP loss-curve evidence collected at 150M, RFC
drafted, experiments/ migration done, Phase 3 adapter designed and
code-drafted. Your job is **Phase 3 execution** on real 8-GPU PP — prove
naive PP numerics match single-GPU reference, then A/B the cross-stage
caching adapter for the "<5% PCIe overhead" headline that carries PR #2.

---

## 2. Two repos — know what's where

- **Workspace logbook**: `git@github.com:QIU023/torchtitan_attention_residual.git`
  branch `main` (HEAD ~= `4abf51d`). Contains ROOT_PLAN, RFC draft,
  `phase2/` + `phase3/` playbooks and prior results. **NOT** tracked by
  torchtitan; this is the project diary + launch scripts.
- **Fork**: `git@github.com:QIU023/torchtitan.git` branch
  `attention_residual_dev` (HEAD ~= `bfe200e`). All AttnRes code lives
  under `torchtitan/experiments/attn_res/`.

Clone both as **peers**:

```
~/work/
├── workspace/                    # from torchtitan_attention_residual
│   └── phase3/go_8gpu.sh         # <-- orchestrator entry point
└── torchtitan/                   # from QIU023/torchtitan
    └── torchtitan/experiments/attn_res/   # all PR code
```

---

## 3. Commandments (violate at peril)

### 3a. Don't modify torchtitan core

The fork already has a migration commit (`976132f → bfe200e`) that
moved AttnRes out of `torchtitan/models/` into
`torchtitan/experiments/attn_res/`. **Do not revert this.** All new
code changes must stay inside `torchtitan/experiments/attn_res/` or
`workspace/phase3/`. The precedent is `experiments/transformers_modeling_backend`
which customizes PP via `ModelSpec.pipelining_fn` without touching core.

If you think you need to modify `torchtitan/distributed/pipeline_parallel.py`
or `torchtitan/models/*` core — stop, flag to user, propose an
experiment-local workaround first.

### 3b. Bugs we already fixed (don't re-hit them)

- **FSDP DTensor bug**: `layer.forward_attn_res()` called directly
  bypasses FSDP pre-forward `all_gather` hook → params stay DTensor →
  `rms_norm` mul crashes. Fix: `AttnResLlama3TransformerBlock.forward`
  dispatches via `__call__` with a kwarg that routes to the
  `forward_attn_res` body. See `experiments/attn_res/model.py`.
- **HF datasets streaming httpx crash**: mid-run `Cannot send a request,
  as the client has been closed` killed our N=12 ablation at step 8810.
  **Mitigation**: pre-fetch C4 shards with `phase3/prefetch_c4.py`
  **before** any long training run on the rental box. Do not skip this.
- **CheckpointManager not on by default**: torchtitan's
  `CheckpointManager.Config.enable` defaults to `False`. The experiment's
  `config_registry.py` now sets `enable=True, interval=1000,
  keep_latest_k=3` — so a mid-run crash can resume. If you run without
  this enabled, you're flying without a net.

### 3c. The one private API we rely on

`pipeline_llm_with_cache_adapter` walks `pp_schedule._stages` (or
`._stage`) to wrap each stage's `.submod`. These are private torch
attributes. If a future torch release breaks this, the
`_iter_schedule_stages` helper in `experiments/attn_res/pipeline_adapter.py`
fails loudly with a clear message. Don't silence the error, fix the
iteration.

---

## 4. What's proven on the current single-GPU run

Llama3-150M dense (actual 75.5M params with tied embeddings), BF16 FSDP,
C4-en streaming, 20k steps. Same-step train loss delta (AttnRes better):

| step | baseline | attn_res (N=6) | Δ |
|---:|---:|---:|---:|
| 500 | 6.141 | 6.015 | −0.127 |
| 5000 | 4.358 | 4.270 | −0.088 |
| 10000 | 4.324 | 4.219 | −0.104 |
| 15000 | 3.737 | 3.686 | −0.051 |
| 20000 | 3.685 | 3.619 | **−0.066** |

Num_blocks ablation at 150M:

| N | final loss | Δ vs baseline |
|---|---:|---:|
| — (baseline) | 3.685 | — |
| 3 | 3.655 | −0.030 |
| 6 | 3.619 | **−0.066** |
| 12 | (in-progress locally, retry after httpx crash) | — |

Throughput: baseline ~71k tps, attn_res ~49k tps (−30 %). Memory +1 GiB
(matches paper's O(Nd) claim with N=6, d=768).

Under PP the tps delta should narrow substantially (block stacking
gets split across stages; each stage only sees its own blocks).

---

## 5. What you're about to do

### Step 0 — boot the box
Activate the python env (likely `/venv/main` on vast.ai images, or a
conda env you create). Verify `torchrun`, `nvidia-smi`, and 8 GPUs
visible. Clone both repos as peers per Section 2.

### Step 1 — run the orchestrator
```bash
bash ~/work/workspace/phase3/go_8gpu.sh
```

What this does:
1. Sanity-checks GPU count & env
2. `pip install -e ~/work/torchtitan[dev]` (idempotent)
3. Downloads Llama-3.1 tokenizer from NousResearch mirror
4. Prefetches 150 C4 shards (~45 GB, ~22B tokens of runway) —
   dominant time cost, 10-30 min depending on bandwidth
5. Runs the AttnRes unit tests (must be 14/14 PASS)
6. **Naive PP smoke**: `PP=8, VP=2, FSDP inner, 150M AttnRes,
   500 steps, adapter OFF`. Loss curve must track Phase 2's single-GPU
   curve within bf16 tolerance.
7. **Adapter PP smoke**: same, `TORCHTITAN_ATTNRES_CACHE=1`, adapter
   ON. Loss must match naive within bf16 tolerance; per-stage send
   size should become constant in stage id (measure via `nsys profile`
   or `torch.profiler`).
8. Calls `compare_pp_vs_single.py` to print max-abs-diff.

Total wall: ~45–60 min if everything works. First run almost certainly
won't — expect 1-2 config issues (layer-per-stage divisibility,
FSDP+PP compose, `parallelize_llama`'s FQN expectations for AttnRes
subparams). Debug on the fly; do not retreat to single-GPU.

### Step 2 — after the A/B data is in, decide
- Loss match + constant send size ⇒ move to the scale-up headline run
  (Llama3 1.5–2B, 20B tokens). Need to add a new config in
  `experiments/attn_res/config_registry.py`.
- Loss diverges under adapter ⇒ one of the 5 open unknowns in
  `phase3/adapter_design.md` is biting. Document what you saw;
  flag to user. Naive PP (no adapter) still produces a valid, if
  weaker, PR #2 story; that's the fallback.

---

## 6. Key files you should read before touching anything

**In the fork (at `~/work/torchtitan/`):**
- `torchtitan/experiments/attn_res/README.md` — experiment overview
- `torchtitan/experiments/attn_res/attn_res.py` — the `block_attn_res`
  primitive (pure torch, no distributed)
- `torchtitan/experiments/attn_res/model.py` — `AttnResLlama3TransformerBlock`
  and `AttnResLlama3Model` subclasses. Note `_return_only_new_blocks`
  flag: when set by the adapter, intermediate-stage forward returns only
  blocks committed by THIS stage (not cached prefix).
- `torchtitan/experiments/attn_res/pipeline_adapter.py` — the
  `CrossStageCacheAdapter` class + `pipeline_llm_with_cache_adapter`
  pipelining_fn. Read the docstring.
- `torchtitan/experiments/attn_res/__init__.py` — model flavors
  (debugmodel_attn_res, 150M_attn_res, 150M_attn_res_n{2,3,4,12}) +
  model_registry wiring.
- `torchtitan/experiments/attn_res/config_registry.py` — Trainer
  configs. Checkpointing is on (`enable=True, interval=1000`).
- `torchtitan/experiments/attn_res/tests/test_attn_res.py` — 14 CPU
  unit tests. Run first before anything expensive.

**In the workspace (at `~/work/workspace/`):**
- `phase3/README.md` — staging plan
- `phase3/adapter_design.md` — **read this if the adapter misbehaves**;
  covers state machine, invariants, 5 open unknowns (microbatch keying,
  VP chunk order, backward-hook reliability under `PipelineScheduleMulti`,
  AC interaction, FSDP reshard composition)
- `phase3/go_8gpu.sh` — the orchestrator
- `phase3/launch_8gpu_naive.sh` / `launch_8gpu_adapter.sh` — individual
  stage launchers; called by `go_8gpu.sh`
- `phase3/prefetch_c4.py` — C4 shard prefetch
- `phase3/compare_pp_vs_single.py` — post-hoc TB comparison
- `phase3/fake_pg_test.py` — single-GPU fake-PG smoke; **optional**,
  use only if you want to reproduce local numerics outside 8-GPU
- `RFC_DRAFT_v2.md` — draft RFC; do not publish yet, user hasn't okayed
- `ROOT_PLAN.md` — full original plan incl. hardware decisions and risk
  register

---

## 7. What you MUST NOT do without asking user

1. Publish the RFC to github.com/pytorch/torchtitan Issues (user hasn't
   okayed; wait for 8-GPU data first).
2. Force-push over the fork or workspace repo without `--force-with-lease`
   and a clear commit reason in chat.
3. Rent more machines. You're on one.
4. Delete any `runs/` directory, even "crashed" ones; they're evidence.
5. Upgrade torch or any torchtitan dep unprompted — the Phase 2 numbers
   were generated on a specific torch version, reproducibility matters.
6. Modify `torchtitan/` core files (Section 3a).

---

## 8. User-specific working preferences

- Terse Chinese-language updates in chat; don't over-explain.
- Report problems with root cause + proposed fix, not just "it failed."
- For long-running commands, use `run_in_background: true` or `Monitor`
  so the user can continue work in parallel.
- Commit and push results after every meaningful milestone so the
  workspace repo stays a live logbook.
- If a step is risky or irreversible (force-push, rm -rf, git reset
  --hard), propose first, act after explicit OK.

---

## 9. Ready check

Before you start, verify (in a read-only way):
- `nvidia-smi -L | wc -l` is 8
- `torchrun --version` works
- You can `git clone git@github.com:QIU023/torchtitan.git` (ssh
  access)
- You have at least ~100 GB free on the disk (env + cache + runs)
- `$HF_HOME` points to a big disk (recommend setting to
  `/workspace/hfcache` or wherever the big disk is mounted)

When all green, run:
```bash
bash ~/work/workspace/phase3/go_8gpu.sh
```

and tell the user what happened.

---

## 10. One-sentence summary

**Run the orchestrator, A/B naive vs adapter PP at 150M, report back,
do not modify core.**
