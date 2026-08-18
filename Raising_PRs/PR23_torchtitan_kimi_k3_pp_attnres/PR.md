# PR #23 — Kimi K3: pipeline parallelism with Block Attention Residuals

**Target**: `pytorch/torchtitan`, on top of #4025 and PR22 (TP)
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
