# Everything we post on the upstream K3 eager PR — consolidated, final

Supersedes and merges `REVIEW_4025_DRAFT.md`, `REVIEW_4025_ARCHITECTURE_COMMENT.md`,
`REVIEW_4025_FINDINGS_2026-08-04.md` (deleted). The RFC #3029 update body lives in
`RFC3029_UPDATE_2026-08-04.md` and is NOT restated here.

**Venue plan**: substance (matrices, branches, qualifications) goes to #3029.
On the PR: ONE short top-level comment + ONE review bundling all inline
comments (a single review = a single notification). Tone: questions and
offers, never verdicts. No PR/issue numbers in commit messages, ever.

---

## 1. Top-level comment (short)

> Thanks for landing this -- an eager reference is the right first step.
> We have been building the parallelism side of K3 (TP/EP/PP/CP, text and
> multimodal) in a fork since release and would like to converge on your
> module layout rather than carry a parallel one -- details and evidence in
> the RFC #3029 update, so as not to flood this thread. Four branches are
> ready to rebase onto this folder once it lands. Two config-level questions
> below, plus a review with interface asks -- all "leave a seam", none
> "change the math".

## 2. Architecture question A: final layer is not global attention

> Should `full_attention_layers` include the last layer? Sec 2.1 places an
> extra Gated MLA at the end of the backbone "ensuring that the final layer
> always performs global attention", and 93 = 23*4 + 1 lines up with that.
> As written, `{4, 8, 12}` over 13 layers ends the stack on KDA. Raising it
> at config level because a 2.8T config would need 92 and 93 both global,
> which the "every (ratio+1)-th layer" pattern cannot express.

## 3. Architecture question B: final aggregation over block representations

> Is the final aggregation over the N block representations intended to be
> out of scope? Sec 2.2 has the per-layer attention (eq. 10) and then "The
> final output layer then aggregates all N block representations"; I only
> see the former here. Happy to contribute it if wanted.

## 4. Review — inline comments (one review submission)

**`model.py`, decoder loop (THE ask — decides patch-vs-fork for PP):**

> Could the decoder forward be factored so the layer loop runs over an
> arbitrary contiguous range and takes/returns its carried state? Block
> AttnRes threads a stack of committed block residuals alongside the hidden
> state; a PP stage owns a layer slice, so the loop must be enterable at
> layer i with `(x, block_residuals)` and exitable at j returning the pair.
> Welded to "all layers, hidden state only", PP means duplicating the body.

**`state_dict_adapter.py`:**

> Room for a hook letting a backend declare per-expert <-> grouped-GEMM
> expert layout? The released checkpoint is per-expert; grouped kernels want
> stacked. A hook keeps one adapter for both directions.

**`parallelize.py`, the unsupported-parallelism guard:**

> If support lands piecemeal (TP before CP), would you take the guard
> per-feature instead of one list? It is the first thing every follow-up
> PR would touch.

**`distributed/fsdp.py`, `add_zero_valued_dependency` (agree + two data points):**

> Strong agree with this helper. Two data points from the same architecture
> under more parallelisms: (1) it is not only "a batch with no images" --
> under CP a rank's shard can hold zero vision sentinels while every rank
> got images, same failure mode at cp>1 only; (2) the failure is a hang, not
> an error -- worth saying in the docstring, a hang sends people to the
> wrong place. Happy to contribute a cp>1 regression test.

**`multimodal.py` consumers, CP design (filling a stated gap, not a bug):**

> qwen3_5 and kimi_k2_7 turn CP off noting multimodal CP needs the vision
> scatter before CP shards. We have multimodal CP working with the opposite
> split -- shard first, each CP rank selects its encoder-output slice via a
> prefix sum over per-rank sentinel counts -- keeping the change model-side.
> That needs `get_vision_positions` to accept a shard (global token count +
> rank offset, absent/truncated runs valid). Would you take that, or prefer
> scatter-before-sharding? Happy to implement either.

**`model.py`, dtype consistency (minor):**

> `pixel_values` casts to `patch_embed.weight.dtype` on the real path but
> `_encode_placeholder_image` uses the activation dtype. They coincide under
> bf16 training.dtype but diverge under fp32 params + FSDP mixed precision,
> and then two DP ranks feed different dtypes into the same module. Cheap to
> derive both from one expression. (Not run end-to-end; the two lines just
> disagree.)

**`model.py`, KDA gate dtype (question, not defect):**

> Was bf16 storage for `A_log`/`dt_bias` considered? fla's kernel
> accumulates their grads in fp32 then downcasts to the param dtype, and we
> measure those grads 2-3 orders below the model median, so bf16 quantizes
> them. No convergence difference measured, and fp32-param recipes are
> unaffected -- but note FSDP2 needs uniform dtype per group, so "just fp32
> them" needs its own group (we tried the naive version; FSDP2 rejects it).

**`config_registry.py`, naming (one-liner):**

> We keep `kimi_linear_*` for the published Kimi-Linear-48B / Table-2 rows
> and `kimi_k3_*` for K3 proper (no K3 exists at 48B). If you have a
> convention for the shared folder we will follow it; cheaper now than in a
> rebase.

## 5. NOT raising (and why)

- The per-layer vs per-sublayer AttnRes count (eq. 8/10): both trees share
  the two-per-layer reading; unsettled, and raising it presents our own
  design as their problem.
- Our PP cross-stage adapter as a generic mechanism: declined upstream
  before; stays private in the model folder.
- Any change to their eager forward math.

## 6. If the thread turns adversarial: what we checked and is fine

SiTU-GLU both branches (beta 4.0/25.0, fig. 4); full-rank Gated MLA gate
(eq. 7); AttnRes pseudo-query Linear(dim,1) + RMSNorm keys (eq. 8); 3:1
KDA:MLA, block size 12; core CrossEntropyLoss. Saying so makes the rest read
as review, not complaint.
