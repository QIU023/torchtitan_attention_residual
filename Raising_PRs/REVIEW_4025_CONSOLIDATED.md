# Everything we post on the upstream K3 eager PR — final

**STATUS 2026-08-05: POSTED AND ADOPTED.** Both review comments landed as
commits on the PR within a day: comment 1 as "Close the Kimi K3 backbone on
global attention" (`9b9fe1feb`, docstring cites the released
`..., 88, 92, 93` list), comment 2 as "Note the context-parallel route in
add_zero_valued_dependency" (`9f60a3d6d`). This file is now a record. The
one remaining unposted piece is the multimodal follow-up response below --
post it when the parallelism follow-ups come up on the thread.

**Venue plan**: substance (matrices, branches, qualifications) goes to the RFC update (`RFC3029_UPDATE_2026-08-04.md`). On the PR: ONE top-level comment (intro + the numbered summary below) + ONE review bundling the two long versions as inline comments at their file locations — a single notification. Tone: questions and offers. No PR/issue numbers in commit messages, ever.

**Format note**: paste sections are written as single-line paragraphs with no blockquote markers, so they copy into GitHub verbatim. Everything between the BEGIN/END markers is the paste.

---

## Top-level comment

--- PASTE BEGIN ---

Thanks for landing this -- an eager reference is the right first step. We have been building the parallelism side of K3 (TP/EP/PP/CP, text and multimodal) in a fork since release and would like to converge on your module layout rather than carry a parallel one -- evidence and branches in the RFC #3029 update, so as not to flood this thread. Two review comments below:

1. **Final layer** (config question): the released config's `full_attn_layers` ends `[..., 88, 92, 93]` -- 92 and 93 both global, per sec 2.1; `{4, 8, 12}` over 13 layers ends on KDA.
2. One field note on `add_zero_valued_dependency`: cp>1 reaches the same deadlock by a different route (a shard with zero vision sentinels on a batch that does have images); our CP follow-up carries a regression test for it.

Neither changes the eager math.

--- PASTE END ---

## Review comment 1 — anchor at `__init__.py` (`full_attention_layers`)

--- PASTE BEGIN ---

Should `full_attention_layers` include the last layer? The released `config.json` lists `full_attn_layers` explicitly as `[4, 8, 12, ..., 84, 88, 92, 93]` -- 24 entries, with **92 and 93 both global** -- which matches sec 2.1's "an additional Gated MLA layer is placed at the end of the backbone, ensuring that the final layer always performs global attention". As written here, `{4, 8, 12}` over 13 layers ends the stack on KDA.

Raising it at config level rather than as a debug-model nit: the released list is not expressible as "every (ratio+1)-th layer", so a 2.8T config needs either the trailing layer appended or the list carried verbatim.

--- PASTE END ---

## Review comment 2 — anchor at `distributed/fsdp.py` (`add_zero_valued_dependency`)

--- PASTE BEGIN ---

Strong agree with this helper. One field note from running this architecture under CP: the trigger is not only "a batch that happens to carry no images" -- under context parallelism a rank's sequence shard can hold zero vision sentinels even when every rank received images, which reaches the same subset-collective deadlock by a route that only appears at cp>1. Might be worth a sentence in the docstring, since the CP route is easy to miss when reading from the DP example. Our CP follow-up includes a regression test for exactly this route.

--- PASTE END ---

## Follow-up response (post as a reply on the thread when appropriate)

--- PASTE BEGIN ---

One addition on scope: the multimodal path (the MoonViT tower + K3 backbone, vision features scattered into placeholder runs) is also already running under PP, EP and CP on our fork -- same matrix, same seeds, results in the RFC update linked above. Stacking those onto this folder may need minor changes on top of the current interfaces -- the shape we saw was things like the PP split needing to see through the multimodal wrapper, and modality-dependent collectives being decided from the mesh rather than from the batch -- nothing structural. Happy to discuss and review together when those follow-ups open.

--- PASTE END ---

---

## Dropped from the posting set (2026-08-04, and why)

- **Decoder-loop seam — MOVED to the RFC update**: we will add both seams
  (initial-stack argument + layer-range slice) ourselves as the PP PR's
  first commit, no dependency on anyone. Announcing our own plan belongs on
  our own thread, not in their review; the PR comment set stays pure review.
  No rebase offer -- we are adding it either way.
- **Unsupported-parallelism guard — DROPPED, same self-reliance logic**:
  the first of our follow-up PRs to land restructures the guard while
  removing its own entry; a 5-line list is not worth a pre-ask.
- **Expert state-dict layout hook — MOVED to the RFC update**: the EP
  branch already carries the complete conversion (hf_key_map, bidirectional
  per-expert <-> grouped with sliced-tensor indexing, plus the grouped
  experts module) -- asking for a hook would hand the design to a thread
  whose author is already asking maintainers whether to build grouped-GEMM
  MoE himself. Announce ours on our thread; ownership follows the
  implementation, not the suggestion.
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
