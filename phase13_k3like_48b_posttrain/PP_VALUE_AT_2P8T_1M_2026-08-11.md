> **The PP conclusions in this file were revised mid-document and are SUPERSEDED by
> `PP_DECISION_2026-08-11.md`.** That file computes it once, separating full-parameter
> from QLoRA and bandwidth from memory. What stays valid here is the mechanism, the
> measurements, and the scaling arithmetic; what to act on is the other file.

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

**Correction (2026-08-11).** An earlier version of this line said "PP for full-parameter
post-training at <= 32k context". That states a conclusion about `m` as though it were about
context length, and it is wrong in both directions. The variable is

    m = tokens_per_step / (tokens_per_microbatch x dp_shard)

and a single 1M causal sequence cannot be split into microbatches, so context length
constrains `m` only through how many WHOLE sequences a step's token budget holds. Two
consequences the old framing hid:

* **Long context does not rule PP out.** At 1M with `dp_shard = 1`, 8M tokens/step gives
  `m = 7.6` and an 19% bubble at pp8 x vp4; 16M gives `m = 15` and 10%.
* **Coding-agent finetune is usually not "many 1M sequences".** Agent traces and repo
  contexts are a long-tailed distribution, and document packing rebuilds `m` from shorter
  documents. At 4M tokens/step and `dp_shard = 2`: ctx 8k gives `m = 244` (0.7% bubble),
  32k gives 61 (2.8%), 128k gives 15 (10.3%), and only a genuine 1M-per-document workload
  collapses to `m = 2` (48%). So the question to ask a workload is the MEDIAN document
  length, not the advertised maximum context.

PP is still wrong for LoRA (the memory argument is gone, the bubble remains) and
structurally wrong for GRPO (small `m` by design).

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

## A worked config: GB300 NVL72 + IB, 1M context, full parameter, tp=1

Assumptions, stated so they can be corrected: 72 GPUs per NVLink domain, 288 GB HBM,
~1.8 TB/s NVLink and ~100 GB/s IB per GPU, and the published 2.78T / 104.2B activated.

**The first thing the arithmetic says is that PP does not reduce per-GPU state at all.**
State sharding degree is `pp x cp x dp_shard = N` whichever way the factors fall:

| N | state per GPU |
|---|---|
| 576 (8 racks) | **87 GB** |
| 1152 (16 racks) | **43 GB** |

So the usual memory argument for PP is void here. What `pp` actually changes is **where the
FSDP all-gather happens**, because torchtitan flattens `dp_shard x cp` into the FSDP shard
dim -- so at `pp=1` that mesh is the whole cluster and every weight gather crosses IB:

| N | pp | cp | dp | FSDP mesh | in one rack? | dense gather / step | AttnRes cache | bubble (vp4, m=4) |
|---|---|---|---|---|---|---|---|---|
| 576 | 1 | 96 | 6 | 576 | no | 1600 ms | 5 GB | 0% |
| 576 | 2 | 96 | 3 | 288 | no | 800 ms | 9 GB | 5.9% |
| 576 | 4 | 48 | 3 | 144 | no | 400 ms | 37 GB | 15.8% |
| 576 | 8 | 24 | 3 | **72** | **YES** | **11 ms** | **149 GB** | 30.4% |
| 1152 | 1 | 96 | 12 | 1152 | no | 1600 ms | 5 GB | 0% |
| 1152 | 4 | 96 | 3 | 288 | no | 400 ms | 19 GB | 15.8% |

`pp = 8` is the configuration that puts one stage per rack and keeps every FSDP gather on
NVLink -- a 145x reduction in gather time. It is also where the AttnRes cache costs 149 GB.

**Retraction.** An earlier version of this section said "Block AttnRes is what makes the
topologically correct choice unaffordable". That is the wrong way round. **The cache is what
makes AttnRes work under PP at all** -- without it a later stage either cannot see earlier
blocks, or each rank's cost stops being constant in depth. What the table shows is the
design's inherent memory cost at scale, not a fault in it.

### The cost is inherent, not a representation artifact -- checked against the math

`block_attn_res` computes

    K       = norm(V).float()                      # V = stack(earlier blocks + current partial)
    logits  = einsum("d,nbtd->nbt", w_L, K)        # w_L is the CONSUMING layer's pseudo-query
    weights = softmax(logits, dim=0)               # normalised over the block axis
    h       = einsum("nbt,nbtd->btd", weights, V)

Two properties follow, and they close off the obvious optimisations:

* **The mixing weights depend on the blocks themselves** (through `norm(V)`) **and on the
  consumer's own `w_L`** -- not on the consumer's hidden state. So a producer stage cannot
  pre-reduce for downstream consumers: it does not hold their `w_L`. The block bodies have
  to travel.
* **Backward needs every `V_i` individually**: `dL/dV_i` is `alpha_i * dh` plus the path
  through `K = norm(V_i)` into the logits. So a flash-style online-softmax accumulator --
  which would let the forward keep one running `[B, T, D]` instead of N blocks -- saves
  FORWARD memory only. Backward still needs them, and recomputing them is impossible because
  the producing layers' weights and activations are on other stages.

So the per-block retention is required. The question is only *where* those bytes live.

### Offload is the real answer, and it is topology-specific

The cache's lifetime is exactly offload-shaped: written once at the end of a microbatch's
forward, read once in its backward, untouched in between.

| | 37 GB (pp4) | 149 GB (pp8) |
|---|---|---|
| PCIe 5.0 x16, ~50 GB/s | 1.5 s round trip | **6.0 s** |
| NVLink-C2C to Grace, ~900 GB/s | 0.08 s | **0.33 s** |

Against a step whose compute is ~7 s, 0.33 s is absorbable and 6.0 s is not. **So this
mitigation exists on GB300/GH-class hardware and does not transport to x86 + PCIe**, which is
worth saying plainly rather than presenting offload as a general answer. On a discrete
accelerator behind PCIe the same 149 GB is simply unavailable, and the honest conclusion
there is `pp <= 2` or no PP.

### The grad bridge has no offload path either, and it is a SECOND block-sized store

Checked: there is no `offload`, `pin_memory`, `.cpu()` or `non_blocking` anywhere in
`pipeline_adapter.py`. Everything is device-resident.

And `_captured_grads[key] = grad.detach().clone()` keeps a full `[B, T, D]` clone per
`(mb, producer_stage, block_idx)` that a later same-rank virtual stage read. During backward
both stores are live, so the figures above are the FORWARD cache only and the true peak is
higher by the same-rank subset. Any offload design has to cover both halves, and the grad
half is harder because its access is interleaved with backward rather than bulk.

### A cheaper, independent win: the fp32 transient

`K.float()` and `V.float()` materialise the whole block stack in fp32. For one microbatch at
pp8/cp24 that is 4.7 GB bf16 + 9.3 GB + 9.3 GB, so the aggregation's transient is several
times the per-microbatch cache. fp32 is load-bearing (the pseudo-query gradient shows 6x-15x
cancellation, measured), but the einsum can be chunked over the block axis and keep fp32.
That is a much cheaper change than offload and helps every configuration.

### Recommendation

**Primary: `pp = 1`, `cp = 96`, `dp_shard = N/96`, EP as large as divides `cp x dp_shard`.**
State fits (87 GB at 576, 43 GB at 1152), the cache is negligible (5 GB), there is no
bubble, and the 1.6 s of gather is prefetch-overlappable against a step whose compute is on
the order of 7 s at 12.6M tokens (6 x 104.2e9 x 12.6e6 FLOPs against ~1e18 FLOP/s
cluster-effective). The one thing to insist on is that **the EP mesh be laid out inside a
rack**; expert all-to-all at 896 experts is the other big IB consumer and it does not
overlap as forgivingly.

**Fallback if measurement shows comms-bound: `pp = 2`.** It halves the per-rank gather for a
5.9% bubble and 9 GB of cache. Cheap insurance, and the only PP degree that is cheap.

**`pp >= 4` at 1M needs cache offload first**, and only on hardware where offload is cheap.
On GB300 that is a 0.33 s round trip against a ~7 s step and `pp = 8` then becomes the best
configuration rather than the most expensive one -- it is the only one that keeps FSDP
gathers on NVLink. On PCIe-attached accelerators the same offload is 6 s and the answer stays
`pp <= 2`.

**Prerequisite for changing that answer:** the cache mitigation already listed in
`PP_ADAPTER_AT_2P8T` -- store only the blocks a LATER virtual stage on this rank actually
reads, which the layout knows. If that cuts the cache by the factor it looks like it should,
`pp = 8` becomes the best configuration rather than the most expensive one.

### On TP, since it was asked

**No, and not merely "later".** `cp x tp` shares the 96-head budget, so every factor of TP
costs a factor of CP -- and at 1M context CP is what keeps both activations and the AttnRes
cache affordable (the cache is inversely proportional to `cp`). TP's one unique win here is
sharding the 163840-wide logits, and chunked loss already covers that; the price of chunked
loss is that MTP does not work with it today (finding 44), which is an MTP problem rather
than a TP argument.

TP becomes the right trade at SHORT context, where CP is not needed and the head budget is
free. At 1M it is competing for exactly the resource that makes 1M work.
