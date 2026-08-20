# ViT run-ahead:为什么与气泡互斥

从 `kimi_k3/vit_prefetch.py` 的模块 docstring 搬出。原文 43 行。内容未改。

Measured earlier, three ways, that this cannot be done by making the ViT a pipeline
STAGE: the bubble count is identical with and without that (2019 either way), every
IR reorder is rejected by the lowering as unschedulable, and the reason both hold is
that a stage sits in the dependency chain -- at one vision stage it IS the pipeline
head, so the bubbles are downstream of the work that would fill them. Overlap
therefore has to come from concurrency, not from placement.

## Why the hook is on the STEP and not the stage

PP hands the first stage one micro-batch's inputs at a time, so at micro-batch m the
stage cannot see m+k's pixels. But ``kwarg_mbs`` -- the per-micro-batch kwargs list --
is passed to ``schedule.step()`` whole. Capturing it at step entry makes every
micro-batch's vision input available from the first action onward, with no change to
core: the hook goes on the schedule instance from our own ``pipelining_fn``.

## The two constraints that are not negotiable

**Same Python thread.** The AttnRes adapter keys its per-micro-batch cache in a
``threading.local``, and its ``forward`` reads a missing key as "this call is PP's
shape inference" and diverts WITHOUT raising. Driving any model code from a worker
thread would silently return shape-inference outputs. Concurrency comes from a CUDA
stream; the prefetch is issued from the same thread that runs the schedule.

**Collectives gated on the mesh, never on the data.** The vision encode issues
collectives (dynamic CP's gather-KV, the feature all-gather, FSDP's tower
all-gather). Every rank in those groups must reach every one of them in the same
order, or two communicators deadlock on a cyclic wait without either one's ordering
being violated. So the prefetch schedule is a function of the micro-batch COUNT --
which every rank agrees on before the step -- and never of what a rank's own batch
contains.

## Memory is the bound on the lookahead

Holding k micro-batches of vision features live is the cost, and it is what decides
whether "most" of the encoder can be hidden or only some. ``depth`` defaults to 1 for
that reason: one micro-batch of slack is enough to overlap an encode with a text
forward, and every extra unit is paid for in resident activations.
