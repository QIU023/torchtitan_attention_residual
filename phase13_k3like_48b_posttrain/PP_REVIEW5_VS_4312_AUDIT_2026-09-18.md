# pp_review5 against PR 4312's head, and the 6840 audit re-checked (2026-09-18, Windows side)

Superseded the same evening: `pp_review5` was reset on the fork to `3f6f35127`,
which is the PR head replayed onto main `a3a819c67` (`ab8ea5f61`, 37 commits)
plus the two commits that ship the recomputing aggregation, and the objects this
note called missing now all fetch. Everything below about the old
`6042863a4` tip is history. The audit of what actually shipped is in
`MEGATRON6840_VS_4312_GAPS_2026-09-18.md`, section "Windows-side audit of the
shipped form". What survives from this note is the discipline it was written
for: read the fork's refs before auditing a branch by name, because the names
here get reused.

Asked for while the GPU box re-implements and smokes: the commit diff between
`pp_review5` and 4312's head restricted to what touches Kimi K3 and the
pipeline, and a re-check of the conclusions already written down.

Everything below was read from refs that exist on the fork or from the installed
torch. Where a document's figure could not be reproduced from those, it is named
as such rather than repeated.

Corrected the same day (user, 2026-09-18): the first version of this note read
the core transport on `pp_review5` as a re-proposal of the mechanism the
maintainer refused in April. It is Elfie's multi-node hang fix, opt-in behind an
environment switch, on the branch built for her two-node test, and it was
dropped after she re-ran multi-node on the 4312 tree without a hang. The
provenance paragraph below carries the reading that holds, and the reason to
work on the PR's line instead is the branch's age.

## The refs, as they actually are

    origin/pp_review5   6042863a4   author date 2026-09-09, 19 commits over base d398a8fb9 (main of 2026-09-10)
                                    78 commits behind upstream/main 68c97b0c5
    4312 head           de6f29514   author date 2026-09-14, also origin/pp_review4 and origin/k3_pp_text
    commits only on pp_review5      19
    commits only on the 4312 head   50

Three objects the two documents rest on are not on the fork: `291f3844a` (the
gaps note's "current pp_review5"), `ab8ea5f61` (the rebase note's replay of the
PR head onto `a3a819c67`) and both `pp_review5_*_20260917` tags. `git ls-remote`
returns one ref for the name, `refs/heads/pp_review5 -> 6042863a4`. The related
local object `c6cea394e` is an ancestor of neither line (23 ahead of
`pp_review5`, 70 behind it) and no remote branch contains it; its 23-commit
spread is the number the gaps note attributes to "the current pp_review5", so
that note describes a line that lives only on the GPU box.

This matters for the instruction that the PP work was checked out to
`pp_review5` and rebased there: on the fork, `pp_review5` is the older line, not
the PR's.

## The diff that answers the question

K3 and pipeline paths only, `pp_review5` to `de6f29514`: 17 files, +768/-631.

    torchtitan/distributed/pipeline_parallel.py                 357   core, see below
    torchtitan/models/kimi_k3/parallelize.py                    251
    tests/unit_tests/cpu/test_kimi_k3_pp_exact_block_grads.py   +232   only on the PR head
    tests/unit_tests/cpu/test_kimi_k3_pp_layout.py              119
    tests/unit_tests/cpu/test_kimi_k3_stage_swap.py              +84   only on the PR head
    torchtitan/models/kimi_k3/layout.py                          80
    torchtitan/models/kimi_k3/model.py                           80
    torchtitan/models/kimi_k3/pipeline_stage.py                  68
    torchtitan_recipes/tests/b200.py                             +37   only on the PR head
    torchtitan/config/configs.py                                 +19
    the rest                                                     under 20 each

The one that decides how this reads: `pp_review5` adds 314 lines to core's
`torchtitan/distributed/pipeline_parallel.py`, and the 4312 head adds 3 and
defines nothing there. What `pp_review5` puts in core is a whole neighbour-P2P
transport: `_PipelineTransportGroups`, `_NeighborP2PTransportMixin` overriding
`get_fwd_recv_ops`, `get_bwd_recv_ops`, `get_fwd_send_ops` and
`get_bwd_send_ops`, plus `_send_meta` / `_recv_meta`, the class factory
`_neighbor_p2p_stage_class`, `_configure_neighbor_p2p_schedule` and
`_warmup_pp_edge_communicators`.

Provenance, corrected after the first version of this note read that code as a
re-proposal of the generic cross-stage mechanism. It is not. `pp_review4` plus
two commits is what `PP_TRANSPORT_NOTE_FOR_ELFIE.md` describes: the edge
communicator warm-up (`fd7ff7400`) and Elfie's multi-node NCCL hang fix composed
onto `AttnResPipelineStage`, with the transport behind
`TORCHTITAN_PIPELINE_NEIGHBOR_P2P=1` so a two-node run could tell the two
remedies apart. `pp_review5` is that line rebased onto main `d398a8fb9` on
2026-09-10 and handed to her to test on two GB200 nodes. The branch
`k3_pp_transport` (`8126172f8`) carries the original port, 407 lines in the same
file.

She then re-ran multi-node on the 4312 tree and saw no hang, so the whole core
change was dropped: the PR head defines none of those identifiers, and the
pipeline work lives inside `kimi_k3/` through `AttnResPipelineStage`,
`layout.py` and `parallelize.py`. The 314 lines on `pp_review5` are therefore a
record of a finished experiment, not a live proposal, and they are evidence that
the branch predates the decision rather than a reason to avoid it.

Identifier presence, both lines, `torchtitan/` only:

    identifier                            4312 head   pp_review5
    _kimi_k3_num_stages                        yes         no
    _as_attn_res_stage                         yes         no
    _swap_in_attn_res_stages                   yes         no
    PPRankLocalCache                           yes         no      (named RankStore there)
    layer_to_stage_from_split                  yes         no
    AttnResPipelineStage                       yes        yes
    kimi_k3_module_fqns_per_model_part         yes        yes
    _NeighborP2PTransportMixin                  no        yes
    _PipelineTransportGroups                    no        yes
    _neighbor_p2p_stage_class                   no        yes
    _warmup_pp_edge_communicators               no        yes

So the PR head is not a superset of the fork's `pp_review5`, and the fork's
`pp_review5` is not a subset of it. `module_fqns_per_model_part` appears four
times in `configs.py` on the PR head against two on `pp_review5` and two on
upstream, which is the exclusivity check the gaps note credits the PR head with.

The two commit subjects that note names as only on `pp_review5` (the config
exclusivity of `module_fqns_per_model_part` against
`pipeline_parallel_layers_per_stage`, and the LLM split taking modules pinned to
the first and last stages) match no commit on either fork line.

## The 6840 conclusions, re-checked

Held, verified here:

- Gap 1 exists on the real PR head. `model.py:165` builds `values_TND` with
  `torch.cat((block_residual_TND, partial_block_TD.unsqueeze(1)), dim=1)` inside
  a ternary starting at 162, and `model.py:247` re-concatenates the stack per
  block start; `pipeline_stage.py:83` and `:102` are the two `torch.stack` calls.
  The note's `167` and `251` are the same two sites numbered on its own line.
- Gap 3 does not port, and its reasoning is the right one. In the installed
  torch 2.13.0, `torch.distributed.pipelining._backward.stage_backward` does
  `grad_inputs.append(val.grad)` and then `val.grad = None`, so the `.grad` slot
  6840's tap and its view-based unpack rely on is cleared under
  `torch.distributed.pipelining`. The `detach().requires_grad_(True)` at
  `pipeline_stage.py:86` is load bearing, as the note says.
- Gap 3's arithmetic reproduces exactly: 2048 tokens by 7168 in bf16 is 28.00
  MiB a slice, 93 layers at block size 12 is 8 blocks, and the comparison rows
  are 28.0 / 112.0 / 224.0 MiB for k = 1 / 4 / 8 against 336.0 (12 slices),
  448.0 (16) and 224.0 (8).
- The AC-reuse reading is right. PR 4656 is two commits over its base, and the
  one that matters adds a `remat.checkpoint` wrapper plus a `checkpoint_residual`
  flag without touching `_apply_attention_residual`'s body, so a 6840-style
  rewrite of that body would not collide in text while excluding it in effect.

Needs a qualifier:

- The surface count for item 1's wide form (`sharding.py`, 3 references) is read
  from a rebased tree, not from the PR. `sharding.py` exists on all three refs,
  but `block_residual` appears in it only on upstream/main, where #4499's tensor
  parallel landed on 2026-09-16, two days after `de6f29514`. On the PR head and
  on `pp_review5` the count is zero. The note's conclusion is unaffected and in
  fact strengthened: the wide form waits until after the rebase, because that is
  when the SPMD declarations it would cost exist.
- The section on what `pp_review5` carries that the PR branch does not reads
  "close to nothing". That is true of the 23-commit local line it measured and
  false of the fork's `pp_review5`, which carries the 314 core lines above.

Not checkable here:

- Every 6840-side figure. The workspace's `Megatron-LM` checkout has no
  `megatron/core/transformer/attention_residual.py` and no ref matching 6840, so
  the 770 lines, the line numbers 663 and 346, the quoted `_AttnResAggregation`
  docstring and the +12.0% to +27.1% padding measurement can only be confirmed
  where they were read. Nothing here contradicts them.

## What follows

- Port onto the PR's own line, not onto the fork's `pp_review5`. The branch that
  backs 4312 is `k3_pp_text` = `pp_review4` = `de6f29514`; `PR_BODY_PP.md` has
  said so since 2026-09-11 and the rebase note repeats it as the lesson of the
  day.
- The reason not to build on the fork's `pp_review5` is its age, not its
  transport: it predates 50 commits of the PR's own line, so it lacks the
  exact-block-gradient and stage-swap tests, the two b200 pipeline cells, the
  `RankStore` to `PPRankLocalCache` rename and `layer_to_stage_from_split`,
  `_kimi_k3_num_stages`, `_as_attn_res_stage` and `_swap_in_attn_res_stages`. Work
  ported there would have to be replayed onto the PR line anyway.
- Item 1 narrow form is the one to take: `_apply_attention_residual` takes a
  sequence and its three call sites pass the stored sources plus the partial,
  which removes the per-sub-layer concatenation inside `model.py` only, with no
  contract change and no SPMD change.
- The rebased head `ab8ea5f61` is worth pushing somewhere before more work lands
  on it, since it currently exists only in a worktree on the box and every
  document that cites it points at an object nobody else can fetch.
