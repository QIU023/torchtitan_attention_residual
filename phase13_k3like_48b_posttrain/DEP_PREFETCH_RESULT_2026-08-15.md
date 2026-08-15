# DEP vision prefetch: pp8xvp4 gate met, and what it does and does not show

Branch `dep_exp_impl`. Gate as specified: pp8 x vp4 passes on the multimodal and the
LoRA flavor, and the ViT's encodes are demonstrably issued ahead of the text pipeline
rather than inline.

## Result

| run | loss lines | prefetch, per step |
| --- | --- | --- |
| mm baseline (prefetch off) | 10 | -- |
| lora baseline (prefetch off) | 10 | -- |
| mm depth 1 | 10 | 7 hit / 1 miss, 27.45 ms over 7 encodes |
| lora depth 1 | 10 | 7 hit / 1 miss, 24.10 ms over 7 encodes |
| mm depth 2 | 10 | 7 hit / 1 miss, 27.29 ms |
| lora depth 2 | 10 | 7 hit / 1 miss, 23.91 ms |

Loss parity, prefetch off against on, step by step: **identical on all four pairs**.
The prefetch changes when the encode runs, not what it computes, and that is now
measured rather than argued.

Seven of the eight micro-batches per step are served by an encode issued during an
earlier micro-batch's text compute. The single miss is the first micro-batch, where
nothing can have been prefetched yet -- so the hit rate is at its structural ceiling.

**depth 2 buys nothing.** Identical counters to depth 1, which is the useful negative
result: the ceiling is the first micro-batch, not the lookahead, so the module's
default of 1 is right and extra depth only costs resident activations.

## What this does not show

Hits prove the encode was ISSUED AHEAD and served from cache. They do not prove it
OVERLAPPED with text compute. Hiding is a claim about when that 27 ms of encode lands
relative to the text stream, and answering it needs a profile or a step-time
comparison, not a hit counter. This box is 8x RTX 5060 Ti over PCIe, so its
communication fraction does not extrapolate to the report's fabric either.

So the defensible statement is: the concurrency the report describes runs here, is
numerically inert, and covers 7 of 8 micro-batches. Not: most of the vision compute
is hidden.

## How this relates to the earlier negative result

`VIT_DEP_DESIGN_2026-08-07.md` refuted the OTHER route -- making the ViT a pipeline
stage. Its measurement stands: 2019 bubbles with and without DEP, vision work in a
compute slot rather than a bubble, and `register_custom_function`'s closed set of
computation types forbidding a `VIT_FORWARD` action. Both remain true.

What changed is that the concurrency route was already built and simply never
enabled: `vit_prefetch` defaults to 0 and no gate had ever set it. Its two hard
constraints were solved at design time rather than worked around --

* concurrency from a CUDA stream on the SAME Python thread, because the AttnRes
  adapter keys its per-micro-batch cache in a `threading.local` and reads a missing
  key as "this call is PP's shape inference", so a worker thread would silently
  return shape-inference outputs;
* the prefetch schedule a function of the micro-batch COUNT, which every rank agrees
  on before the step, never of what a rank's own batch contains -- otherwise two
  communicators can deadlock on a cyclic wait without either one's ordering being
  violated.

The adapter's thread-local cache therefore never had to be redesigned. That was
recorded here earlier as an open question and it was not one.

## Next, in order

1. **A step-time comparison** at a scale where the tower's cost is visible: the
   overhead to hide is proportional to the tower against the text model, and at the
   debug scale it is 0.12% of a pipeline stage. This is the measurement that would
   turn "issued ahead" into "hidden", and it is the one the proportional-scaling
   instinct was reaching for.
2. **The prefetch in the standing gate**, so it cannot regress unnoticed -- it has
   run once, on two cells.
