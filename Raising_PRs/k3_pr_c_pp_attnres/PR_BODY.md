# PR-C: pipeline parallel for Kimi K3 with Block Attention Residuals

## What

Pipeline parallelism for K3, including Block AttnRes, whose residual stack is a
second value carried alongside the hidden state across stage boundaries.

**BLOCKED on an interface.** #4025's decoder loop iterates all layers from a
single entry point. PP needs it enterable at layer `i` with
`(x, block_residual_TND)` and exitable at `j` returning the same pair. Without
that seam this PR duplicates the loop body, which is not a reviewable state.
The ask is filed on #4025; this branch is ready to rebase the moment it is
answered.

## Design

The cross-stage adapter is **private to the model folder's parallelize** -- the
generic mechanism was proposed upstream before and declined, and we are not
re-proposing it.

Communication is cache-based and incremental: each hop ships only the blocks
this stage newly committed, the receiver rebuilds the full stack from its cached
prefix plus the delta, and entries are released when the microbatch finishes.
That is the same shape the report describes for Block AttnRes under PP.
Non-Interleaved1F1B schedules fall back to the plain path, correct without the
cache saving.


## The local grad bridge, and why it is a hook rather than an autograd.Function

The subtlest part of the adapter, and the part a reviewer should look at first.

A block that stage R commits on this rank during an earlier virtual stage can be read back
from the shared cache by a LATER virtual stage on the SAME rank. Left alone, that later
stage's backward traverses into stage R's forward graph through the rebuild's stack/cat
grad path and FREES it; stage R's own backward, driven later by PP's SEND_B on the outgoing
delta, then dies with "backward through the graph a second time".

Two earlier designs did not hold. A process-global `retain_graph=True` monkey-patch worked
but leaked graphs for every unrelated backward in the process. Wrapping the hand-off in an
`autograd.Function` (`_LocalCacheAugment`) also failed under real scheduling
(Interleaved1F1B + FSDP + selective AC rerun): returning a view of the input did not stop
autograd walking from the consumer's node upstream into `Augment` and on into the
producer's forward graph during the CONSUMER's backward. That traversal was observed firing
the producer-side `Augment.backward` inside the consumer stage's `backward_one_chunk` --
exactly the freeing the design was meant to prevent.

What holds severs the link structurally, in two rank-local halves with no collectives and
no cross-rank state:

* a tensor grad hook on the producer block, which fires once during the producer stage's
  backward, pops the matching captured-grad slot, and sums it into the incoming grad;
* `_LocalCacheCapture` on the consumer side, whose tensor input is a DETACHED leaf taken
  from the cache. **The detach is the load-bearing guarantee**: even if autograd ignored
  Capture's `None` grad return, there is no upstream `grad_fn` left to traverse.

Recv-originated cached blocks -- sliced from a prior `recv_delta` tensor -- are deliberately
NOT detached and NOT wrapped. Their gradient already flows back to the producing RANK
through PP's built-in SEND_B via the recv-delta autograd chain, and severing that link
would strand the cross-rank gradient channel.

## Multimodal

The PP split cannot see through a multimodal wrapper: core's `_split_module`
walks only top-level `named_children()`, so neither flat nor dotted FQNs reach
the text stack and every stage ends up with zero parameters. This PR splits the
text model and rebuilds the wrapper around the chunk that kept `embed_tokens` --
vision features are spliced into the embeddings, so nothing vision-side crosses
a stage boundary. That matches the report's DEP framing, where the tower is not
sharded across stages.

## Verification

PP8 x VP4 (32 layers, `layers_per_stage 1`): |Dloss| 0.0018 vs the no-PP
reference at step 1. Text legs `pp2`, `fsdp2_tp2_pp2`, `tp2_pp2_cp2`,
`fsdp2_pp2_cp2`, `ep2_fsdp2_tp2_pp2`, `ep2_fsdp2_pp2_cp2`; multimodal
`mm_fsdp2_pp2`, `mm_fsdp2_tp2_pp2`, `mm_fsdp2_pp2_cp2`, `mm_ep2_fsdp2_pp2`,
`mm_ep2_fsdp2_tp2_pp2`, `mm_ep2_fsdp2_pp2_cp2`.

## Relationship to #4025

Built on #4025's `torchtitan/models/kimi_k3/` layout. Does not change the eager
forward path. Rebase onto that PR's landing before review.

## Verification (continued)

Seed 42, `--debug.deterministic`, 10 steps unless stated. Full matrices and the
per-defect history are in the RFC and in
`phase13_k3like_48b_posttrain/`.

## Honesty

2.8T has never been run by us; verified on 48B-real-weight shapes and
K3-faithful downscales, scale-out is config-level.
