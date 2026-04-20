# RFC Design Log

A chronological record of the decisions behind the torchtitan AttnRes RFC
(`RFC_DRAFT_v3.md`), extracted from the Phase 2/3 design conversations.
Useful as context for reviewers asking "why did you choose X?" and for the
follow-on Claude session iterating on the 8-GPU side.

---

## Scope decision: two PRs, not one

The RFC and the implementation were split into two PRs from the start.

- **PR #1** — `torchtitan/experiments/attn_res/` with the primitive, the
  Llama3 subclass, unit tests, and single-GPU FSDP evidence. Ready.
- **PR #2** — the cross-stage caching adapter benchmarked on 8×5090 PCIe
  PP=8, plus a 1–2 B scale-up run. In flight on the rental box.

Rationale: a single monolithic PR holding up algorithm correctness on
pending 8-GPU pipeline debugging would sit in review indefinitely. Split
so the primitive + evidence land first and the distributed story is a
focused review of its own.

## Placement decision: experiments/, not core

Initial Phase 2 commit (`2d4d5df → eafe5bc`) added AttnRes under
`torchtitan/models/common/` and `torchtitan/models/llama3/`. That
violated two rules we later re-read carefully:

- `torchtitan/experiments/README.md` principle 3 & 5: experiments
  "should reuse existing torchtitan code as much as possible" and
  dependencies flow one-way (experiments depend on core, not the
  reverse).
- `.claude/CLAUDE.md` core-principle #4: "don't leak experiments into
  core."

Survey of the four parallelism-adjacent experiments (`autoparallel`,
`graph_trainer`, `ft`, `transformers_modeling_backend`) confirmed:
none modify `torchtitan/distributed/`. The canonical pattern —
`transformers_modeling_backend` — provides a custom
`pipelining_fn` in its `ModelSpec` and calls core's `pipeline_llm`
unchanged. We mirrored that.

Result: full migration in a single amended commit
(`eafe5bc → 144d10c → 976132f → bfe200e`). Zero core modifications.
All AttnRes-specific wiring lives in
`torchtitan/experiments/attn_res/`. The `AttnResLlama3Model` and
`AttnResLlama3TransformerBlock` subclass the core blocks; the adapter
enters via `pipeline_llm_with_cache_adapter` in the experiment's
`__init__.py:model_registry`.

## Length decision: short, not comprehensive

First draft (`RFC_DRAFT_v2.md`) was ~1 650 words, ten sections,
multiple code blocks, multiple tables, non-goals, ownership, and a
closing "signals to accelerate" paragraph.

Reference: [pytorch/torchtitan#2536](https://github.com/pytorch/torchtitan/issues/2536)
is ~350 words, zero code blocks, one comparison table, six tight
sections (Problem / Solution / What it enables / Comparison /
Target use cases / Reference). No non-goals paragraph, no ownership
paragraph, no benchmark numbers — posted at the *design alignment*
phase, not the *final-evidence* phase.

Rewrote to v3 at 479 words. Dropped:

- Non-goals / Ownership sections (too defensive for a torchtitan RFC).
- Multi-code-block design sketch (reviewers read the branch directly).
- "Signals that would accelerate this" closing paragraph.
- Long motivation backstory.

Kept:

- Problem (standard residuals dilute shallow signal).
- Solution (Block AttnRes, one-paragraph algorithm).
- Placement decision (experiments/, one paragraph).
- Evidence table (same-step delta milestones).
- Plan (PR #1 ready, PR #2 in flight).
- Open questions for maintainers (3).
- Reference (paper + repo + fork branch).

## Evidence decision: single-GPU Phase 2 is sufficient for the RFC

Initially considered gating the RFC on 8-GPU PP data. Reasoning ran
through several loops and landed on: an RFC exists to align direction
with maintainers *before* the large run, not to report finished
numbers. #2536 posted with no loss curves; our draft posts with a
delta table plus an N-ablation, which is already denser than #2536.

Holding the RFC until full 1–2 B scale-up data would delay reviewer
feedback by ≥ a week. The cost of posting early — maintainer says
"redesign X" after we've already built X — is bounded by how much of
PR #2 is already written. If a redesign request arrives, the
experiments/ structure means the rewrite is contained inside
`experiments/attn_res/`.

## Evidence included

Llama3 dense, 12 layers, ~75 M params with tied embeddings, BF16
FSDP, C4-en streaming, 20 k steps, identical config across variants.
Final delta vs baseline (3.685):

| N (num_blocks) | final loss | Δ |
|---:|---:|---:|
| — (baseline) | 3.685 | — |
| 3 | 3.655 | −0.030 |
| 6 (primary) | 3.619 | −0.066 |
| 12 | 3.623 | −0.061 |

Finding worth noting in the RFC: N=6 and N=12 are essentially tied
at L=12, and N=12 (one layer per block, maximum granularity) does
NOT degrade. The paper reports "N ≥ 16 degrades" at larger L; at
shallow scale the sweet-spot region is wide. This is a small but
honest observation that signals the author understood the sensitivity
surface rather than blindly inheriting the paper's N=8 default.

## PP adapter story as stated in the RFC

Reviewer-facing framing evolved twice.

**v1** (fake-PG first, then 8 GPU): "we will validate numerics first on
single-GPU fake process group PP=4, before burning multi-GPU time."

**v2** (direct 8 GPU): dropped the fake-PG step after the orchestrator
and prefetch scripts were tested. v2 read: "Benchmark plan on 8×5090
PCIe (intentionally PCIe, not NVLink — the cheap/wide-deployment
regime)." Rationale: fake-PG tests the model tuple return but not the
integration with real NCCL; a 500-step sanity on 8×5090 catches the
same bugs in similar wall-time, and directly produces the "before"
number for the adapter A/B.

**v3** (honest status): after the 8-GPU side reimplemented the adapter,
the first cut using `torch.autograd.Function` for grad send-back proved
brittle under `PipelineScheduleMulti` recomputation. Variable-shape
per-stage activations also don't fit
`torch.distributed.pipelining`'s assumption of fixed tensor shapes
between stages. The RFC now says:

> Standard `torch.distributed.pipelining` assumes a fixed activation
> tensor shape across stages, but Block AttnRes's per-stage send
> payload is `(partial, new_blocks_committed_this_stage)` where the
> second tensor's leading dim grows with `stage_id` (naive path) or is
> constant but matched across stages under the adapter. A first cut
> using `torch.autograd.Function` for grad send-back proved brittle
> under interleaved 1F1B recomputation, so the adapter is being
> reimplemented around a custom effective-PP path that does explicit
> NCCL P2P outside autograd, keyed on integer (microbatch,
> producer_stage, block_idx) tags.

This replaces the earlier open question "does `register_hook` survive
`PipelineScheduleMulti`?" — which is now answered in the negative —
with the more useful open question: **"variable-shape activations
between stages — any precedent in torchtitan / `torch.distributed.pipelining`
beyond bypassing built-in P2P?"**

## Open questions currently in the RFC

Three, chosen to be things maintainers can productively answer without
access to our implementation:

1. Is the `pipelining_fn` + `schedule._stages` walk the canonical way
   to wrap stage submodules? We rely on a private torch attribute.
2. Variable-shape activations between stages — see above.
3. VP chunk keying: `(microbatch_id, virtual_stage_id)` vs
   logical-depth block index?

Deliberately omitted as open questions (because they're resolved or
not novel to this work):

- Adapter placement (resolved: `experiments/`).
- dtype policy (paper runs BF16 end-to-end; we match; nothing
  controversial).
- Activation checkpointing interaction (we'll test, not blocking).
- FSDP reshard composition (we'll test, not blocking).

## Files that carry RFC-adjacent artifacts

| Path | Role |
| --- | --- |
| `RFC_DRAFT_v3.md` | Current draft (post-worthy, awaiting user OK) |
| `RFC_DRAFT_v2.md` | Kept for reference; longer first draft |
| `RFC_ISSUE_DRAFT.md` | Earlier pre-Phase-2 sketch |
| `phase2/runs/ablation/comparison.png` | 4-way loss curve plot, the "money shot" for PR #1 |
| `phase2/plot_ablation.py` | Re-runnable ablation plotter |
| `phase2/runs/attn_res/train.log` | Raw log for the N=6 primary run |
| `phase2/runs/ablation/llama3_150m_attn_res_n{3,12}/train.log` | Ablation raw logs |
| `torchtitan/experiments/attn_res/README.md` | In-repo experiment overview (linked from RFC) |
| `torchtitan/experiments/attn_res/pipeline_adapter.py` | The adapter being discussed in §PP-story |

## Meta: when to actually post

Resolution after the full conversation:

- Post conditions met: algorithm works (Phase 2 loss delta),
  experiments/ placement decided, adapter design sketch exists,
  open questions are substantive.
- Post conditions NOT met only by the strictest interpretation:
  "production benchmark numbers." Reference torchtitan RFCs (#2536,
  others surveyed) do not require this.
- Recommendation to the author: post v3 now, reply to reviewer
  questions with "PR #2 in flight, benchmarks by end of week."

The 16-layer full-PP adapter loss curve is NOT a gate on the RFC. It
is a gate on PR #2.
