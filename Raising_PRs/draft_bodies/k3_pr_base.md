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
