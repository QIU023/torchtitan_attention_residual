# PR body: pipeline parallel for Kimi K3

按 maintainer 要求:terse,无小标题,op 级事实在前,数字内联。

---

Adds pipeline parallelism to kimi_k3. The part that is not mechanical is Block
Attention Residuals: a block residual is defined over the whole stack, so it has
to travel between stages as a second stage payload alongside the hidden states.
A stage that dropped it trains a different model -- measured on the debug flavor
at pp2, dropping it moved step 3 from 7.44679 to 9.30017.

Three things in the current tree fail silently without this, all fixed here.
The block residual did not cross stages at all. The final aggregation
(output_res_proj/output_res_norm) ran on every stage instead of only the one
that owns the head. And the FQN injection returned early on a Config-tree model
and said nothing, so the split silently fell back to the generic one and the
last stage lost the aggregation modules.

The strongest check is not a loss curve: splitting the model in two by hand and
running the halves in sequence, with no schedule, no loss and no micro-batches,
reproduces the unsplit forward at max_abs 0.000e+00.

The vision tower needs a stage assignment or core's _split_module sets it to
None on every stage, and the first multimodal batch then reports "pixel_values
were provided without a vision encoder". It rides with whichever chunk kept the
embedding, since vision features are spliced into the embeddings and nothing
vision-side crosses a boundary.

Also here is the report's 5.2.3 DEP clause 1: the tower gets a pipeline stage of
its own, ahead of the text stages, so its compute is off the critical path of the
stage that owns the embedding. It is opt-in because it changes the stage count.
Verified as a pure scheduling change: at dp2 x pp4 from a shared seed
checkpoint, DEP off and on give the same step-1 and step-2 loss, 12.49453, with
one shared warm compile cache and each cell run twice.

Clause 2, splitting the tower across several stages, is not in this PR. It needs
the stage split to address vision_encoder.layers, and _split_module descends only
into a ModuleDict or ModuleList that is a direct child of the model. Asking for
it raises with that reason rather than silently running clause 1. The tower's own
share decomposition is here and is unit-tested bit-identical to the unsplit tower
at 2, 3 and 4 shares, so clause 2 has something to build on.

Files: new pipeline_adapter.py, attn_res.py, layout.py, knobs.py, the dep_bubble
trio and dep_vision_stage.py; model.py carries the block residual in and out of a
stage; parallelize.py and __init__.py wire pipelining_fn.

Tested: CPU tests for the FQN injection and the tower share decomposition; pp2,
pp4 and pp8 run on the debug flavor.
