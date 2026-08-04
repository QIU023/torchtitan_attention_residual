# Everything we post on the upstream K3 eager PR — final

**Venue plan**: substance (matrices, branches, qualifications) goes to the
RFC update (`RFC3029_UPDATE_2026-08-04.md`). On the PR: ONE top-level
comment (intro + the numbered summary below) + ONE review bundling the long
versions as inline comments at their file locations — a single notification.
Tone: questions and offers. No PR/issue numbers in commit messages, ever.

---

## Top-level comment (post as-is: intro + numbered summary)

> Thanks for landing this -- an eager reference is the right first step.
> We have been building the parallelism side of K3 (TP/EP/PP/CP, text and
> multimodal) in a fork since release and would like to converge on your
> module layout rather than carry a parallel one -- evidence and branches in
> the RFC #3029 update, so as not to flood this thread. Short version, long
> versions in the review:
>
> 1. **Decoder loop (the one that matters for PP)**: could the layer loop be
>    factored to run over a contiguous range, carrying `(x, block_residuals)`
>    in and out? Details in the review on `model.py`.
> 2. **Final layer** (config question): sec 2.1 puts an extra Gated MLA at
>    the end so the final layer is global attention; `{4, 8, 12}` over 13
>    layers ends on KDA.
> 3. **Expert state-dict layout hook**: a seam for per-expert <->
>    grouped-GEMM conversion, so one adapter serves both layouts.
> 4. **Unsupported-parallelism guard**: per-feature instead of one list, so
>    partial support lands without editing every entry.
> 5. Two field notes on `add_zero_valued_dependency` (cp>1 reaches the same
>    failure mode; the symptom is a hang) + a regression-test offer.
>
> None of these change the eager math -- all "leave a seam".

## Long versions (ONE review; anchor each at the named file)

**1. `model.py`, decoder loop — decides patch-vs-fork for PP:**

> Could the decoder forward be factored so the layer loop runs over an
> arbitrary contiguous range and takes/returns its carried state? Block
> AttnRes threads a stack of committed block residuals alongside the hidden
> state; a PP stage owns a layer slice, so the loop must be enterable at
> layer i with `(x, block_residuals)` and exitable at j returning the pair.
> Welded to "all layers, hidden state only", PP support means duplicating
> the body rather than reusing it. TP, EP and CP need no such seam.

**2. `__init__.py` (`full_attention_layers`), final layer:**

> Should `full_attention_layers` include the last layer? Sec 2.1 places an
> extra Gated MLA at the end of the backbone "ensuring that the final layer
> always performs global attention", and 93 = 23*4 + 1 lines up with that.
> As written, `{4, 8, 12}` over 13 layers ends the stack on KDA. Config
> level matters here because a 2.8T config would need 92 and 93 both
> global, which the "every (ratio+1)-th layer" pattern cannot express.

**3. `state_dict_adapter.py`, expert layout hook:**

> Room for a hook letting a backend declare per-expert <-> grouped-GEMM
> expert layout? The released checkpoint is per-expert; grouped kernels
> want stacked. A hook keeps one adapter for both directions.

**4. `parallelize.py`, the guard:**

> If support lands piecemeal (TP before CP), would you take the guard
> per-feature instead of one list? It is the first thing every follow-up
> PR would touch.

**5. `distributed/fsdp.py`, `add_zero_valued_dependency`:**

> Strong agree with this helper. Two data points from the same architecture
> under more parallelisms: (1) it is not only "a batch with no images" --
> under CP a rank's shard can hold zero vision sentinels while every rank
> got images; same failure mode, cp>1 only. (2) The failure is a hang, not
> an error -- worth one docstring sentence, a hang sends people to the
> wrong place. Happy to contribute a cp>1 regression test.

---

## Dropped from the posting set (2026-08-04, and why)

- **Final aggregation question — WITHDRAWN, wrong**: the eager PR has it
  (`output_res_norm/proj` built at model level, applied over the full block
  stack before norm + lm_head; adapter maps the released
  `output_attn_res_*` keys). The audit had only read the block class.
  Now on the checked-and-fine list below.
- **CP multimodal design question** (`get_vision_positions` accepting a
  shard): our shard-first CP is already implemented on the PR-D branch, and
  the helper belongs to PR #3532's multimodal common code, not to this PR.
  Raise it in PR-D's own description where the helper actually changes.
- **dtype consistency nit + KDA gate dtype question**: we matched their
  tree on both this week; the questions are academic until a measured
  difference exists. Kept internal (`KDA_GATE_DTYPE_2026-08-04.md`).
- **Naming convention**: resolved on our side (`kimi_k3_*` for K3 proper,
  `kimi_linear_*` only for the published Kimi-Linear models).

## NOT raising (unchanged)

Per-layer vs per-sublayer AttnRes count (shared reading, unsettled); our PP
adapter as a generic mechanism (declined before; stays in the model
folder); any change to their eager math.

## If the thread turns adversarial: checked and fine

SiTU-GLU both branches (fig. 4); full-rank Gated MLA gate (eq. 7); AttnRes
pseudo-query Linear(dim,1) + RMSNorm keys (eq. 8); the final aggregation
over block representations (sec 2.2, present at the model tail); 3:1
KDA:MLA, block size 12; core CrossEntropyLoss.
