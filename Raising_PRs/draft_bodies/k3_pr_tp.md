Draft. This branch is our Kimi K3 implementation carrying the tensor parallel plan; the other two axes raise in the parallelize entry, so what is under review here is TP and the model it needs.

PR-4025 is a separate implementation of the same model, further along on the model itself and with all four parallelism axes raising `NotImplementedError`. When it lands this rebases onto it and the diff narrows to the TP plan alone. It is a draft now so the axis work is visible while that happens -- and so the three axis PRs can be read as a set, since today each carries the model and they overlap heavily for that reason.

Every Colwise/Rowwise entry in this TP plan uses `use_local_output=True`, so module
boundaries stay plain tensors while the TP collectives still fire inside each Linear.
That is the constraint the whole file is organised around, and it comes from three
places in K3 where DTensor does not compose: fla-core's triton kernels
(`causal_conv1d` inside `ShortConvolution`, and `FusedRMSNormGated`) call into triton
on data pointers and crash on a DTensor argument; AttnRes stacks partial-block and
completed-block tensors, which raises on mixed operands; and PP send/recv only carries
plain tensors.

PR-4025 rejects TP explicitly, citing data-dependent Python loops and incompatible
forward signatures. This is the work that answers that.

The fla shims are bound per instance, on the modules of the model being parallelized,
and marked idempotent. An earlier form assigned `cls.forward` on the fla classes, which
mutates a third-party library process-wide and reaches models that never enabled TP.

Two defects turned up while doing this. `block_attn_res` was requesting `Partial()` on
the tp axis for the pseudo-query gradient, reasoning that K/V carry Partial gradients
back from rowwise projections. That is wrong: grad_query is sum(grad_logits * K) with
both factors replicated on tp, so every rank already holds the whole gradient and
Partial made the backward sum tp identical copies. Per parameter on one dense MLA layer
the two AttnRes projections sat at 0.4999 under tp2 and 0.2501 under tp4, exactly 1/tp,
while the other 18 parameters sat at 1.0000; with the default placement all of them do.
The other, `moe_sharding.py` dropping the computed `in_grad_placements` when EP is off,
reproduces on unmodified deepseek_v3 and is filed separately.

Per-parameter gradient comparison against a shared seed checkpoint, varying only tp:
one dense MLA layer 0.00002 (tp2) and 0.00003 (tp4) max |ratio-1|, one KDA layer 0.0010,
one warm MLA+MoE layer 0.00836 and 0.01518, 21 dense layers plus AttnRes 0.0004.

Ten steps on three model arms -- text, multimodal, multimodal plus LoRA -- with only tp
varied, first and last loss:

    tp2   text 7.72089 -> 4.96644   mm 12.04522 -> 10.20116   mm+lora 12.02691 -> 11.91231
    tp4   text 7.70643 -> 4.93963   mm 12.07788 -> 10.26537   mm+lora 12.08271 -> 11.94110

Those are the tp-only cells. This branch carries no CP or EP plan, so combinations with
them are not what it should be judged on; they exist and are the subject of the sibling
PRs.

tp8 is not reachable here: k3mini has 4 attention heads, and a widened 8-head flavor
then trips "strides should be multiple of 16 bytes" at that shard width. Verified at
tp2 and tp4 only.

One note on why a declarative `sharding_config` cannot simply replace this file, since
it is the obvious question. Declarations cover more than the weight -- `in_src_shardings`
lifts a plain input via `DTensor.from_local`, so a module fed plain activations can be
driven declaratively. What the vocabulary has no field for is the output side of
`use_local_output`: `out_dst_shardings` redistributes, it does not `to_local`, so a
declaratively-driven module always returns a DTensor. K3's residual stream is plain and
Block AttnRes injects two more plain sources into it, so `down_proj` declared rowwise
returns a DTensor and the residual add then mixes kinds. The migration unit is a whole
residual stream rather than a module: migrating the dense FFN alone dies with
`aten.add.Tensor got mixed torch.Tensor and DTensor`, while the 26 layer norms migrate
byte-identically because their output feeds attention instead. `lm_head`, `embed_tokens`
and those norms are declarative already.
