# 气泡运行时:钩子位置与代价核算

从 `kimi_k3/dep_bubble_runtime.py` 的模块 docstring 搬出。原文 36 行。内容未改。

Two hook points look equally reasonable and only one of them can fire where it
matters.

``fwd_recv_ops.pop`` sits immediately before ``_wait_batch_p2p`` in the runtime's
FORWARD branch, which is the moment the rank is about to block on a receive -- the
start of an idle interval, and it carries the ``(stage, microbatch)`` an anchor needs.
That was the first implementation. It fires for a FORWARD that RECEIVES activations,
and the rank owning the tower owns pipeline stage 0, whose input comes from the
dataloader: no receive, no pop, nothing fires. Measured on a real pp8xvp4 cell -- 8
placements planned, 0 fired.

So the anchor moved to the action BEFORE the interval, and the hook to after
``forward_one_chunk`` returns. The interval starts exactly there: that action has
completed and the next cannot begin until a dependency lands. No receive is involved,
so it works on the stage that actually holds the tower.

## Main stream, not the side stream

The prefetch path issues encodes on a side stream so they overlap with whatever the
main stream is doing. This path is the opposite by design: the bubble IS main-stream
idle time, so the encode belongs on the main stream, where it occupies the gap rather
than competing for SMs with text compute.

## What is not here yet

Backward. The report handles the backward passes analogously, and the same hook exists
for it (``bwd_recv_ops``), but the vision backward is triggered by the spliced
features' gradient rather than by a schedule action, so it needs the adapter's gradient
path involved. Forward first, with the occupancy criterion, then backward.
