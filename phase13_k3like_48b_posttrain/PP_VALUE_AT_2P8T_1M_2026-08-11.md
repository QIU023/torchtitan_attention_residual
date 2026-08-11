# Is PP worth it for 2.8T post-training and LoRA at 1M context?

Short answer: **PP and 1M context work against each other**, and LoRA removes most of PP's
reason to exist. The regime where PP earns its keep is full-parameter post-training at
moderate context, not either of the cases asked about.

> **CAVEAT added 2026-08-11.** The per-layer figure below (29.9B) is derived from the
> PUBLISHED 2.78T total divided by 93 layers, which is right for real K3. Our provisional
> `kimi_k3_2p8t` config does NOT reproduce that total: with `num_experts=896` and
> `moe_intermediate_size=3072`, one routed-expert set is `3 x 7168 x 3072 x 896` = 59.2B per
> MoE layer, so 92 MoE layers give **5.45T, about 1.96x the published 2.78T**. The activated
> count was validated against the model card (104.2B); the TOTAL was never checked.
>
> Two shapes would fit 2.78T -- `moe_intermediate 1568` at 896 experts, or **457 experts** at
> moe_intermediate 3072 -- and the released 2.8T `config.json` is not on this box, so which
> field is wrong cannot be settled here. Every memory figure in this document that comes
> from the published total is unaffected; anything measured by RUNNING the 2p8t flavor would
> be roughly 2x too large.

## The numbers this rests on

`kimi_k3_2p8t`: 93 layers, hidden 7168, 896 experts, vocab 163840.

| | |
|---|---|
| params per layer | 29.9B total, **1.12B activated** |
| full-parameter state (bf16 w + bf16 g + fp32 m/v/master) | **44.5 TB** |
| LoRA state (frozen bf16 base) | **5.6 TB**, or **1.4 TB** with the MXFP4 base we already support |

## Why PP loses at 1M context: the bubble is a function of microbatch count

Interleaved 1F1B leaves a bubble fraction of `(p-1) / (v*m + p-1)`. At 1M context each
microbatch IS ~1.05M tokens, so `m` directly sets the tokens per step:

| pp x vp | m | bubble | tokens/step |
|---|---|---|---|
| 8 x 4 | 2 | **46.7%** | 2.1M |
| 8 x 4 | 4 | 30.4% | 4.2M |
| 8 x 4 | 8 | 17.9% | 8.4M |
| 8 x 4 | 32 | 5.2% | 33.6M |
| 16 x 4 | 4 | 48.4% | 4.2M |

To get the bubble under 20% at pp8 x vp4 you need 8 microbatches, i.e. **8.4M tokens per
step**. That is a plausible pretraining batch and an implausible SFT or GRPO one -- RL in
particular wants small, frequent policy updates, so `m` is small by design and the bubble is
where the pipeline goes. Long context and PP are competing for the same knob.

Note the direction: deeper PP makes it worse at fixed `m` (pp16 is worse than pp8), because
the warmup is longer and there are no more microbatches to hide it.

## Why our adapter makes PP worse at 1M specifically

The cross-stage payload is `[seq_local, hidden]` per microbatch per hop, and Block AttnRes
adds the 8 cached blocks on top:

| context | cp | hop | hop + AttnRes cache |
|---|---|---|---|
| 32768 | 8 | 0.055 GiB | 0.44 GiB |
| 262144 | 8 | 0.438 GiB | 3.50 GiB |
| 1048576 | 8 | 1.750 GiB | **14.0 GiB** |
| 1048576 | 64 | 0.219 GiB | 1.75 GiB |

Those are per microbatch, and the cache is held for every microbatch in flight. So the
combination "PP + 1M + AttnRes" is the most memory-hostile configuration this model has,
and it is the one PP is least able to pay for in throughput. CP is what makes it survivable
at all -- cp=64 brings the cache back to 1.75 GiB.

## What PP actually buys, and where

Three things, and only the first is unique to PP:

1. **It divides FSDP's weight-gather traffic by `p`.** Every FSDP rank must materialise the
   full non-expert weights of whatever layer it is computing, and that traffic does not
   shrink by adding data-parallel ranks. Under PP a rank owns `93/p` layers and gathers only
   those. This is a per-step bandwidth win that no other axis provides.
2. **It bounds per-stage state deterministically.** Useful for capacity planning, but FSDP
   already shards state across ranks, so this is a convenience rather than an enabler.
3. **It divides activation depth**, which reduces how hard activation checkpointing has to
   work. Real, but AC can absorb it.

Under LoRA, (2) largely evaporates: there are no optimizer states or gradients for the base,
so state collapses from 44.5 TB to 5.6 TB (1.4 TB quantized), and the frozen base also means
FSDP has no gradient reduce-scatter for it. The memory argument that motivates PP is mostly
gone, while the bubble cost stays.

**So: PP for full-parameter post-training at <= 32k context, where `m` can be large and (1)
pays. Not for LoRA, and not at 1M.**

## If we only had FSDP + EP + CP: the four real limitations

Not fatal, and worth stating precisely rather than as a warning.

### 1. Depth stays on every device, so activation checkpointing becomes mandatory

Without PP, each device holds activations for all 93 layers of its sequence shard. AC stops
being an optimisation and becomes a requirement, and full AC costs roughly an extra forward
pass in recompute. PP would have divided the depth instead of recomputing it. This is the
cost that is paid in FLOPs rather than in memory.

### 2. No vocab sharding, so chunked loss is mandatory -- and that currently excludes MTP

TP is what shards the `[*, 163840]` logits. Without it the only defence is chunked loss,
which we have, and at 1M context it is not optional: the fp32 logits upcast alone is
`seq x 163840 x 4` bytes.

The sharp consequence found this week: **MTP and chunked loss are incompatible today**
(finding 44 -- MTP needs full-vocab logits per depth, and chunking exists so they are never
materialised). So without TP, **MTP and long context cannot coexist in this stack**. That is
a concrete, dated limitation rather than a general worry, and the fix is a per-chunk MTP
loss.

### 3. FSDP's per-layer gather traffic has a floor that more ranks do not lower

Adding data-parallel ranks shards the *storage* but every rank still gathers a full layer's
non-expert weights to compute it. That per-step traffic is fixed by the model, not by the
cluster. EP already removes the expert weights from that gather -- which is why FSDP+EP is
the standard MoE combination -- but the dense/attention remainder across 93 layers is a real
per-step bandwidth cost, and PP is the only axis that divides it.

### 4. No TP means no sequence-parallel relief on the non-attention parts

TP+SP shards norms, residuals and the FFN activations. Without it those stay full-width per
device. CP compensates where it matters most at long context (it shards the sequence, which
is the dominant term at 1M), which is why CP is the right substitute here -- but it is a
substitute, not an equivalent: CP does nothing for the width dimension.

## Recommendation

| regime | stack |
|---|---|
| full-param post-training, <= 32k | FSDP + EP + TP + PP (+VP). PP's traffic division pays, bubbles affordable |
| full-param post-training, 1M | FSDP + EP + CP, PP only if the batch can afford `m >= 8` |
| **LoRA SFT, any context** | **FSDP + EP + CP.** PP adds bubbles and cache pressure for a memory problem LoRA already solved |
| GRPO / RL | FSDP + EP + CP. Small `m` by design makes PP's bubble structural |

Two things to fix if we want the no-PP/no-TP path to be complete: a **per-chunk MTP loss**
(removes limitation 2) and **measured AC recompute cost at depth 93** (quantifies
limitation 1). Neither needs PP.
