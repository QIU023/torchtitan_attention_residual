# PR #23 — Kimi K3: pipeline parallelism with Block Attention Residuals

**Target**: `pytorch/torchtitan`, `main`. Draft.
**Scope**: `pipeline_adapter.py` (1,770 lines) and `layout.py`, the V=1 thin mode, plus the
AttnRes model itself (`attn_res.py`, `attn_res_model.py`) which upstream references by config
in seven places without implementing. DEP -- the vision tower on its own stage, with the
optional bubble scheduling (`dep_bubble_{plan,runtime,backward}.py`, `vit_prefetch.py`) --
rides with this PR because it is the same pipeline machinery, but see the note below on what
it does and does not deliver.

**On DEP, stated plainly so a reviewer is not misled**: the mechanism is complete and
numerically correct -- the tower gets its own stage, the encodes are placeable into the
schedule's idle intervals, and the deferred vision backward is bit-identical to the inline
one. What is NOT established is a speedup. Measured by step time, the bubble scheduling is
NEGATIVE at both shapes that fit in 15.5 GiB per GPU: -0.95% at cost ratio 0.493 (the point
theory calls maximal) and -2.08% at ratio 2.0. The binding constraint turns out to be the
report's own upfront prefix rather than the cost ratio -- the first few micro-batches'
encodes cannot be placed because nothing precedes them to anchor on, so at pp=4 with 8
micro-batches half are unplaceable before the run starts and the ceiling on any gain is 25%.
Placed share is about `(mb - pp) / mb`, which improves only with `mb >> pp`, and a larger mb
is exactly what does not fit at seq 4096 here. So DEP is offered as working machinery whose
benefit needs a bigger box to demonstrate, not as a measured win.
**Risk**: this is the hard one. AttnRes makes PP non-standard, and the adapter is
where that is handled.
**Depends on** the structural ask in #4025 review: factor the decoder loop so it
can run over an arbitrary layer range with `(x, block_residual_TND)` in and out.
That IS pipeline splittability, and it is what a thin shim can then bind to.

## Why AttnRes changes PP

A standard decoder stage takes a hidden state and returns a hidden state. With
Block AttnRes, each layer also reads a stack of completed block representations
and, at block boundaries, appends to it. So a stage boundary has to carry TWO
things -- the hidden state and the block stack -- and the stack grows as the
pipeline advances.

That is not a cache: it is a live autograd path. Gradients flow back through the
block stack across stage boundaries, so the adapter has to route them, not just
transport the values. This is also why LoRA does not exempt it -- skip-edge
gradients are real edges regardless of which parameters are trainable.

## What the adapter does

- injects the reconstructed stack at stage entry and slices out only the newly
  appended segment at exit (`_return_only_new_blocks`), so P2P carries O(N*d)
  rather than the whole history;
- keeps `layers` as a `ModuleDict` keyed by layer id, because the splitter's
  `layer_to_stage` discovery depends on those string keys;
- stacks blocks into a tensor at the boundary, since P2P sends raw tensors and
  DTensor wrappers do not survive.

## Verification

Per-parameter gradient comparison against a non-PP reference sharing the
accumulation structure -- 4 partial sums of batch 1 on every leg, so the
comparison measures the adapter rather than bf16 summation order:

    pp2         max |ratio-1| over 548 parameters   0.00000
    pp4                                             0.00000
    pp2 x vp2                                       0.00000
    pp4 x vp2                                       0.00000

Loss 7.71304 and grad_norm 8.4957 on every leg, matching the no-PP baseline. The
stage split is real rather than a merge artifact: per-rank parameter counts are
312/280 for pp2 and 138/174/168/112 for pp4, unioning to exactly 592.

## Limits, stated

- `pp8` needs dp_shard=1, which on this GPU leaves KDA in fp32 and asks for
  108,160 B of shared memory against a 101,376 B limit. Not measured.
- PP with the multimodal wrapper does not work: the splitter divides
  `model.layers`, which on that model is a property forwarding into
  `.language_model`, so some stages receive a wrapper whose text model has been
  taken away. Text-only PP is unaffected. Teaching the split about the wrapper is
  separate work and is not in this PR.

## PASTE (the body that goes upstream)

---

Draft. This branch is our Kimi K3 implementation carrying the pipeline parallel plan; the other two axes raise in the parallelize entry, so what is under review here is PP and the model it needs.

PR-4025 is a separate implementation of the same model, further along on the model itself and with all four parallelism axes raising `NotImplementedError`. When it lands this rebases onto it and the diff narrows to the PP plan alone. It is a draft now so the axis work is visible while that happens -- and so the three axis PRs can be read as a set, since today each carries the model and they overlap heavily for that reason.

A standard decoder stage takes a hidden state and returns a hidden state. With Block
AttnRes each layer also reads a stack of completed block representations and appends to
it at block boundaries, so a stage boundary has to carry two things and the stack grows
as the pipeline advances. That stack is not a cache but a live autograd path: gradients
flow back through it across stage boundaries, so the adapter routes them rather than
just transporting values. LoRA does not exempt this -- skip-edge gradients are real
edges whatever is trainable.

The adapter injects the reconstructed stack at stage entry and slices out only the newly
appended segment at exit, so P2P carries O(N*d) instead of the whole history. `layers`
stays a `ModuleDict` keyed by layer id because the splitter's `layer_to_stage` discovery
depends on those string keys, and blocks are stacked into a tensor at the boundary since
P2P sends raw tensors and DTensor wrappers do not survive.

This depends on the structural ask in the PR-4025 review: factoring the decoder loop so
it runs over an arbitrary layer range with `(x, block_residual_TND)` in and out. That is
pipeline splittability, and a thin shim binds to it.

Per-parameter gradient comparison against a non-PP reference sharing the accumulation
structure -- 4 partial sums of batch 1 on every leg, so the comparison measures the
adapter and not bf16 summation order -- gives max |ratio-1| of 0.00000 over 548
parameters at pp2, pp4, pp2 x vp2 and pp4 x vp2, with loss 7.71304 and grad_norm 8.4957
matching the no-PP baseline on every leg. The split is real rather than a merge artifact:
per-rank parameter counts are 312/280 at pp2 and 138/174/168/112 at pp4, unioning to
exactly 592.

Ten steps on three model arms -- text, multimodal, multimodal plus LoRA -- first and last
loss:

    pp2   text 7.70844 -> 4.92657   mm 12.07827 -> 9.90401   mm+lora 12.05631 -> 11.89396
    pp4   text 7.70131 -> 4.86812   mm 12.06958 -> 9.80813   mm+lora 12.07972 -> 11.89169
    pp8   text 7.71037 -> 4.81359   mm 12.06663 -> 9.90860   mm+lora 12.02343 -> 11.89581

Those are the pp-only cells. This branch has no TP, CP or EP plan -- all three raise in
the parallelize entry -- so it is the model plus the pipeline path, and the axis
combinations belong to the sibling PRs.

DEP -- the vision tower on its own stage with optional bubble scheduling -- rides along
because it is the same machinery, and it needs stating plainly. The mechanism is complete
and numerically correct: the tower gets its own stage, encodes are placeable into the
schedule's idle intervals, and the deferred vision backward is bit-identical to the
inline one. There is no speedup. By step time the bubble scheduling is negative at both
shapes that fit in 15.5 GiB per GPU, -0.95% at cost ratio 0.493 (the point theory calls
maximal) and -2.08% at ratio 2.0. The binding constraint is the report's own upfront
prefix rather than the cost ratio: the first few micro-batches' encodes have nothing
preceding them to anchor on, so at pp=4 with 8 micro-batches half are unplaceable before
the run starts and the ceiling on any gain is 25%. Placed share is about (mb - pp) / mb,
which improves only with mb >> pp, and a larger mb is what does not fit at seq 4096 here.

Two limits. pp8 needs dp_shard=1, which on this GPU leaves KDA in fp32 and asks for
108,160 B of shared memory against a 101,376 B limit, so it is not measured. PP with the
multimodal wrapper does not work: the splitter divides `model.layers`, which on that
model is a property forwarding into `.language_model`, so some stages receive a wrapper
whose text model has been taken away. Text-only PP is unaffected; teaching the split
about the wrapper is separate work.

## RETRACTED: the DEP hiding numbers above (2026-08-19)

Every DEP hiding figure in this kit was measured with `KIMI_VIT_DEP_STAGES` at its default
of 1, which leaves the vision tower on the same pipeline stage as the text side. That
changes where the features are consumed, and the consumption point is exactly what decides
whether a bubble can serve a micro-batch. So the placed/unplaced split those numbers report
is not the split the report's design produces.

Three claims are withdrawn outright: that the binding constraint is the upfront prefix (at
pp4/mb64 the prefix is 4 of 64 while 56 stay synchronous); that placed share is about
(mb - pp) / mb (placed stayed at 4 as mb went 16 to 64); and that the mechanism needs 60
GiB per GPU (that was without activation checkpointing, which no flavor enables -- with
full AC, pp2 x vp2 at seq 4096 and mb 16 fits locally at 76.86% memory).

Fixed rather than withdrawn: the planner placed at most one encode per idle slot, which
bounded placements by the slot count however small the cost ratio got. Removing that took
pp4/mb64 from 4 to 8, and the new diagnostics say 0 starved / 10 exhausted, so the
constraint is bubble-to-consumption-point timing rather than budget.

Full accounting in `phase13_k3like_48b_posttrain/DEP_MEASUREMENT_RETRACTION_2026-08-19.md`.
DEP should not be described in an upstream PR until item 1 of that document's next steps is
done.
