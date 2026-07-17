# Phase 13 — K3-like 48B post-training (Kimi-Linear base + AttnRes graft)

> Created 2026-07-16, the day Kimi K3 was announced (weights 2026-07-27).
> Goal: seize the K3 moment by building the **earliest torchtitan-based
> distributed post-training of a K3-like model** on a base with real,
> competitive pretraining — NOT our under-trained 447M. Motivation is now
> distributed-framework-layer credibility (public artifact + LinkedIn/Zhihu/X),
> not resume bullets. See [`../K3_RELEASE_IMPACT_2026-07-16.md`](../K3_RELEASE_IMPACT_2026-07-16.md).

## 0. Why this phase, why 48B

Three hard constraints forced the design:

1. **"Validation without distributed infra is meaningless."** The whole point is
   the framework layer. So the phase must actually exercise 5D (FSDP+TP+EP,
   +PP for continued-pretrain), not FSDP-only smoke.
2. **"Won't show a bad benchmark."** Our 447M is under-trained (5.5 tok/param vs
   paper 200) -> flat RL, 12% GQA. A from-scratch downscale-K3 would be equally
   under-trained. **Only a well-pretrained base gives competitive numbers.**
3. **K3 open weights are 2.8T-only** (the entire K2 family ships one size; K3
   likely too). Nobody solo-post-trains 2.8T. So we need a *smaller* member of
   the same architecture family that already has Kimi-grade pretraining.

**`moonshotai/Kimi-Linear-48B-A3B-Base` is exactly that**: real Kimi pretraining,
open weights, and it IS K3's architecture family one scale down (KDA + MLA + MoE).
Add AttnRes -> "K3-like". This is strictly better than waiting for 7.27 to build a
from-scratch downscale-K3 (which would be under-trained -> bad benchmarks).

## 1. Architecture: K3 vs Kimi-Linear-48B (the graft surface)

K3 blog gives component names; exact dims/N come with the 2026-07-27 config.
Kimi-Linear-48B is fully known (HF config.json).

| Axis | Kimi-Linear-48B (open, known) | K3 (blog) | Delta besides AttnRes |
|---|---|---|---|
| Attention | plain MLA + KDA (3:1) | **Gated MLA** + KDA | MLA gains a gate |
| MoE routing | sigmoid grouped-topk (1 group) | **Quantile Balancing** | new routing, no aux-loss heuristic |
| MoE framework | Kimi sparse / stock | **Stable LatentMoE** | new MoE framework |
| Experts | 256 / 8 act / 1 shared | **896 / 16 act** | count + scale |
| Activation | SwiGLU | **SiTU (Sigmoid Tanh Unit)** | new activation |
| Optimizer | Muon | **Per-Head Muon** | per-head variant |
| Quantization | bf16 | **MXFP4 w / MXFP8 a (QAT)** | native low-precision QAT |
| AttnRes | none | yes | (headline; the graft) |
| Scale | 48B / A3B | 2.8T / A50B | scale |

**6 deltas besides AttnRes.** Which can graft onto *pretrained* 48B weights:

| Component | Graft onto pretrained 48B? | Why |
|---|---|---|
| **AttnRes** | YES | zero-init = identity at init, safe |
| **Quantile Balancing** | YES (training-time) | routing method, not a weight structure |
| **Per-Head Muon** | YES (continued/post-train) | optimizer, doesn't touch weights |
| Gated MLA | HARD | changes attention structure; gate must init ~identity + retrain |
| SiTU | NO | weights trained for SwiGLU; swapping activation breaks them |
| Stable LatentMoE | UNKNOWN | framework-level; wait for report |

**Realistic K3-like graft = Kimi-Linear-48B + AttnRes (+ optional Quantile
Balancing routing + Per-Head Muon optimizer).** Covers K3's core novelty at a
scale we can actually train.

## 2. Distributed-infra completeness (audited 2026-07-16 @ `90d85eba3`)

`torchtitan/experiments/attention_residual/kimi_linear/parallelize.py` (1077 lines)
+ `pipeline_adapter.py` (1143 lines).

| Axis | Status | Location |
|---|---|---|
| FSDP2 / HSDP | DONE | `apply_fsdp` (tied-embed bundle, PP-strip aware, EP-nested edp_mesh) |
| TP | DONE | `apply_tp_kimi_linear` (DSv3-style; KDA=NoParallel; MLA colwise/rowwise; MoE gate/shared NoParallel; fla-core DTensor patch; plain-boundary convention) |
| EP | DONE | `apply_ep_kimi_linear` (ExpertParallel all-to-all + edp FSDP nesting) |
| PP | DONE | `pipeline_adapter.py` pipelining_fn (Interleaved1F1B + cache adapter) |
| AC / compile | DONE | shared `apply_ac` / per-layer compile with fla carve-outs |
| **CP** | BLOCKED | `raise NotImplementedError` — fla-core `chunk_kda` lacks ring-recurrence over CP shards (upstream fla-core issue, not ours) |

**5D (FSDP+DP+TP+PP+EP) is code-complete on the model side.** Only CP is missing
and it is upstream-blocked. KDA already reduces seq-parallel pressure, so CP is
low-priority for K3-arch.

**Honest gaps:**
- **Full-5D simultaneous composition (all five >1 at once) is untested at scale.**
  Pressure test validated PP up to 8xVP=4. Individual axes tested; the full
  simultaneous stack needs multi-H200 validation.
- **Post-training via veRL+torchtitan** — model-side `parallelize_fn`/`pipelining_fn`
  are backend-agnostic (work regardless of who calls them). The gaps are the
  integration glue: (1) does veRL's torchtitan engine-worker drive `pipelining_fn`
  for PP, or only `parallelize_fn`? — verify against veRL source; (2) train<->rollout
  weight reshard for AttnRes+KDA+MoE; (3) rollout-side AttnRes serving (SGLang
  overlay or official). At 48B these are avoidable: **FSDP+TP+EP holds 48B on
  multi-H200 without PP**, so post-training runs today via veRL-torchtitan
  FSDP/TP/EP; the PP-post-training gap only bites at true K3-scale.

## 3. The plan

```
base:        moonshotai/Kimi-Linear-48B-A3B-Base   (real Kimi pretraining)
graft:       + AttnRes (zero-init)  [+ optional Quantile Balancing / Per-Head Muon]
continued-PT: torchtitan MAIN Trainer, 5D incl PP (pipelining_fn is wired here),
             few-B tokens on multi-H200, let AttnRes params settle off identity
post-train:  veRL + torchtitan backend, FSDP+TP+EP (no PP needed at 48B)
             - sequence-KD from K3 API (teacher text -> student SFT/seq-distill)
             - GRPO on a verifiable single/multi-task
rollout:     HF-generate first (fastest), then Kimi-Linear SGLang / AttnRes overlay
benchmark:   48B-base vs 48B+AttnRes A/B, competitive with same-scale (~A3B) models
             on the chosen task(s) — pick tasks where KD+RL actually lifts the base
```

### Step order
1. Load Kimi-Linear-48B-A3B-Base into the torchtitan `kimi_linear` model spec
   (weight adapter: HF -> torchtitan state dict; PR15 already validated the
   inference-side load, reuse the mapping).
2. Register a `kimi_linear_48b_a3b_attn_res` flavor grafting AttnRes onto the
   real config (N per paper sweet-spot ~8 until K3's N is known).
3. Continued-pretrain (5D, torchtitan main Trainer) a few-B tokens on multi-H200.
   A/B checkpoint: base vs +AttnRes.
4. veRL post-training (FSDP+TP+EP): sequence-KD from K3 API + GRPO on target task.
5. Benchmark the A/B; write up.

## 4. Upstream merge — split-merge strategy (reviewed 2026-07-17 @ upstream `fbceec07`)

A real `git merge upstream/main` from `90d85eba3` produces **10 conflicts**, and
they split into two very different halves. The merge was reviewed on the Windows
box (torch absent -> can't `pytest`) and aborted; execute the split below on the
GPU box.

### Key finding: upstream rebuilt the RL experiment

Between our fork point and `fbceec07`, upstream **completely restructured
`experiments/rl/`** — added `components/`, `controller`, `environment/`,
`examples/`, `losses/`, `observability/`; moved model registration from
`plugin.py` to `models/vllm_registry.py`; and it is **vLLM-only (no SGLang)**.
Our RL (`actors/sglang_generator`, `opd_trainer`, `grader`, `plugin`, `utils`) has
no upstream equivalent — it is a parallel universe. **Merging the RL half is
fighting upstream's rebuild for zero phase13 value** (phase13 post-trains via
veRL, not torchtitan-experiment-RL).

### Half A — Core / AttnRes (DO merge; low-risk, mostly "our torch-2.9 compat shim vs upstream newer-torch"; the box is torch 2.11)

| File | Regions | Resolution |
|---|---|---|
| `experiments/__init__.py` | 1 | union: upstream list + our `"attention_residual"` |
| `models/common/attention.py` | 2 | take upstream (2.11 has varlen + inductor flag); ALSO add upstream's `VarlenAuxRequest` import; confirm nothing uses our SDPA fallback |
| `models/common/decoder.py` | 1 | union BOTH params: upstream's `attention_masks` + our `return_outputs` |
| `distributed/parallel_dims.py` | 1 | take upstream (2.11 has `_unflatten`); drop our hasattr guard |
| `distributed/context_parallel.py` | 1 | review on box; same compat pattern, likely take upstream |

### Half B — RL experiment (do NOT reconcile; keep ours, we are on veRL)

`plugin.py` (upstream-deleted; our SGLang reg), `actors/utils.py` (upstream-deleted,
+235 ours), `actors/trainer.py` (5 regions), `types.py` (2), `rl/__init__.py` (1).

### Recipe (path-scoped merge on the GPU box)

```bash
cd torchtitan
git checkout attention_residual_dev && git merge --ff-only origin/attention_residual_dev
git fetch upstream
git merge --no-commit --no-ff upstream/main
# Half B: keep ours wholesale (freeze our SGLang RL; phase13 uses veRL anyway)
git checkout --ours -- torchtitan/experiments/rl/ && git add torchtitan/experiments/rl/
# Half A: resolve the 5 core conflicts per the table above, then git add each
# VALIDATE (mandatory — auto-merged core files may be silently broken):
python -c "import torchtitan.experiments.attention_residual"
pytest torchtitan/experiments/attention_residual/tests/ -x
# smoke a 5-step debug-flavor train (catches semantic breakage in the auto-merged
#   distributed/ + models/common/ files that DID NOT conflict textually).
```

**Insight:** upstream RL is still vLLM-only. Our engine-agnostic SGLang generator
remains a real upstream gap (feeds PR #12). Not "scooped" by the rebuild.

Full conflict list (10): Half A = `distributed/{parallel_dims,context_parallel}.py`,
`models/common/{attention,decoder}.py`, `experiments/__init__.py`. Half B =
`experiments/rl/{__init__,types}.py`, `rl/actors/{trainer,utils}.py`, `rl/plugin.py`.
Beyond these, other core files (`distributed/{activation_checkpoint,utils}.py`)
auto-merge textually but still need the smoke test.

## 5. Honest caveats (do not skip these in any writeup)

1. **AttnRes grafted onto a non-AttnRes-pretrained model may show only a modest
   gain.** Zero-init makes it safe but it starts as identity; a short
   continued-pretrain may not fully activate it. The base-vs-+AttnRes A/B is a
   real result regardless, and the **distributed infra is the primary deliverable.**
2. **"Competitive" must be scoped to chosen tasks**, not broad SOTA. Pick tasks
   where sequence-KD (K3 API) + GRPO genuinely lift the 48B base.
3. **This is not a K3 reproduction.** It is a K3-like model (K3's architecture
   family + AttnRes) at feasible scale. We never post-train the real 2.8T; K3's
   actual weights only enter via sequence-KD (teacher outputs from the API).
4. **Full-5D simultaneous composition is unproven** until run on multi-H200.
5. **After 2026-07-27**: reconcile AttnRes N + placement against K3's real config;
   optionally add Quantile Balancing. Base stays Kimi-Linear-48B.

## 6. Public artifact / promotion hook (LinkedIn / Zhihu / X)

The shareable claim, kept honest:

> "torchtitan-native 5D (FSDP/TP/EP/PP) implementation of Block Attention
> Residuals on the Kimi-Linear 48B backbone — a K3-like model — with distributed
> post-training (veRL + sequence-KD from K3). Open, reproducible, and the earliest
> torchtitan K3-family training stack."

Hold the scope line (§5). The value is the **framework-layer artifact**, not a
benchmark-topping model.

## 7. Confirmed timeline (2026-07-17)

### Before 2026-07-27 (start now — nothing here waits on K3 weights)

1. **phase13 main line**: veRL + torchtitan `kimi_linear`, base = Kimi-Linear-48B-A3B
   open weights, graft AttnRes -> continued-PT (5D) -> veRL post-train (FSDP+TP+EP).
2. **Sequence-KD data collection via the Kimi API — available NOW.** Start pulling
   K3 outputs on the target task(s) into an SFT/seq-distill corpus; this needs only
   the public API, no local 2.8T, no weights. Front-load it so the corpus is ready
   when the trainer is.
3. Upstream merge (split strategy §4) on the GPU box; pytest-gate.
4. Base-vs-+AttnRes A/B on a scoped task.

### After 2026-07-27 (weights + config + report drop)

1. **Reconcile** AttnRes N/placement against K3's real config; optionally graft
   Quantile Balancing / Per-Head Muon (per §1 graftability). Base stays 48B.
2. **2.8T-under-limited-resources scenarios**, ranked by feasibility:

| # | Scenario | Feasible on rented multi-H200? | Value |
|---|---|---|---|
| 1 | **Seq-KD via API** (continue) | yes (no local 2.8T) | primary distill signal |
| 2 | **Config/weight reconciliation** — load K3 config + partial-read AttnRes params | yes (cheap; partial load) | fixes N + validates our port |
| 3 | **K3 MXFP4 inference benchmark** — serve 2.8T, compare our two-phase AttnRes overlay | yes, short rental (~1.4TB MXFP4 -> ~18+ H200) | inference-side evidence |
| 4 | **Logit-level OPD** — local 2.8T teacher -> distill into 48B student | stretch (same ~18 H200, longer run) | stronger than seq-KD |
| 5 | **QLoRA on real 2.8T** | marginal (multi-node, expensive) | probably skip |
| 6 | **Small AttnRes as K3 draft model** (speculative decoding) | niche (needs local K3 inference + logits) | skip unless #3 already stood up |

3. **Decision rule:** scenarios 1-2 are cheap and always-do. 3 is the natural next
   rental (inference-only, short). 4 only if the seq-KD signal proves too weak. 5-6
   are opportunistic. **None of these require training the 2.8T** — the phase13
   deliverable (48B K3-like training stack) stands on its own regardless.
