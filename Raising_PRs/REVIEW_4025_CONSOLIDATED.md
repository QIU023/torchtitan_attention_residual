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
> 1. **Decoder loop (heads-up for PP, not a request)**: the loop already
>    carries `(h, block_residual)`; our PP follow-up will add the two missing
>    seams (initial stack as an argument, layer range as a slice) as its own
>    first commit. Flagging now so the shape does not drift -- happy to
>    rebase if you would rather define it yourselves. Details on `model.py`.
> 2. **Final layer** (config question): sec 2.1 puts an extra Gated MLA at
>    the end so the final layer is global attention; `{4, 8, 12}` over 13
>    layers ends on KDA.
> 3. **Expert state-dict layout hook**: a seam for per-expert <->
>    grouped-GEMM conversion, so one adapter serves both layouts.
> 4. **Unsupported-parallelism guard**: per-feature instead of one list, so
>    partial support lands without editing every entry.
> 5. One field note on `add_zero_valued_dependency`: cp>1 reaches the same
>    deadlock by a different route (a shard with zero vision sentinels on a
>    batch that does have images) + a regression-test offer.
>
> None of these change the eager math -- all "leave a seam".

## Long versions (ONE review; anchor each at the named file)

**1. `model.py`, decoder loop — heads-up, not a request:**

> The loop already threads exactly the right pair -- `h_BLD` and
> `block_residual_TND` in and out of every layer -- so PP needs only two
> small seams: `block_residual_TND` is created inside the forward
> (`h_BLD.new_zeros(B * L, 0, D)`) so a stage cannot pass an incoming stack
> in, and the loop always runs `self.layers.values()` to the end. A PP stage
> owns a layer slice and has to enter at layer i with the upstream stack and
> exit at j returning the pair. **Our PP follow-up PR will add both** (an
> optional initial-stack argument and a layer-range slice) as its first,
> separately reviewable commit -- nothing here blocks on you. Flagging it now
> so the forward does not drift into a shape that makes the seam bigger, and
> in case you would rather define it yourselves pre-merge, which we would
> happily rebase onto. TP, EP and CP need no seam at all.

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

> Strong agree with this helper, and the docstring already names the failure
> correctly ("deadlock the step"). One field note from the same architecture
> under more parallelisms: the trigger is not only "a batch that happens to
> carry no images". Under context parallelism a rank's sequence shard can
> hold zero vision sentinels while every rank did receive images, which
> reaches the same missing-reduce-scatter deadlock at cp>1 only. Might be
> worth half a sentence in the docstring, since the CP route is easy to miss
> when reading the DP one. Happy to contribute a cp>1 regression test.

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

---

## Pre-post verification against the PR head (2026-08-04)

`git fetch upstream pull/4025/head` -- head unchanged at the commit our
worktree already had, so the tree read below is current. Verified by reading
the files, not by grepping one class; that is exactly how the withdrawn
aggregation claim got made in the first place.

| item | verdict | evidence |
|---|---|---|
| 1. decoder loop | **STALE, corrected** | the loop already carries `(h_BLD, block_residual_TND)`. The draft said "hidden state only", which a maintainer would disprove by opening the file. Reframed: the real seam is that `block_residual_TND` is built inside the forward and the range is always the full `self.layers.values()` |
| 2. final layer | PASS | `full_attention_layers = {4, 8, 12}`, matched 1-based via `(layer_idx + 1)`, 13 layers, nothing anywhere appends a trailing MLA |
| 3. state-dict hook | PASS | adapter maps `block_sparse_moe.experts.{i}.{w1,w2,w3}` -> `moe.routed_experts.{i}.{proj}`; per-expert on both sides, no grouped-GEMM seam |
| 4. guard | PASS | one `(name, enabled)` tuple list building `unsupported_parallelisms`, then a single raise |
| 5. add_zero_valued_dependency | **half STALE, corrected** | the docstring already says "deadlock the step", so "the failure is a hang, worth a docstring sentence" was telling them something they wrote. Dropped. The cp>1 route is genuinely absent from it and is kept |
| withdrawn: final aggregation | stays withdrawn | `output_res_norm/proj` declared at 646, built at 692, applied at 830 over the full block stack before norm and lm_head. Independently re-confirmed |

Two of five needed correction, and one of those changes what we are asking
for. Worth recording that the first pass of this review was written from
partial reads twice over -- the aggregation claim and the "hidden state only"
claim have the same root cause.
