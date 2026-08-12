# PP for K3 2.8T post-training: the settled answer

**This supersedes the PP conclusions in `PP_VALUE_AT_2P8T_1M` and `PP_ADAPTER_AT_2P8T`.**
Those two documents each revised the answer mid-way, because they mixed three separate
questions: full-parameter versus QLoRA, memory-driven versus bandwidth-driven, and PP in
general versus PP with our AttnRes cache. Computed once here, separated properly.

## The one number that settles most of it

FSDP all-gathers weights per layer, but **EP keeps the expert weights resident and out of
that gather**. Experts are ~97% of K3's parameters, so the gather is only the ~83B dense
part. That is what makes the no-PP path affordable -- not PP, and not TP.

| | total state | dense gather / rank / step | compute / step |
|---|---|---|---|
| full parameter, bf16 | 50.0 TB | **167 GB** | ~9.1 s at N=576 |
| QLoRA, MXFP4 base | 1.39 TB | **42 GB** | ~54.7 s at N=64 |

Gather time by fabric, per rank per step:

| | NVLink (1.8 TB/s) | IB (100 GB/s) | EFA/RoCE (50 GB/s) |
|---|---|---|---|
| full parameter | 0.09 s | 1.67 s | 3.34 s |
| QLoRA | 0.02 s | 0.42 s | 0.83 s |

## Answer 1: how much is PP worth, per scenario

### QLoRA SFT: **zero, and it is actively negative**

State is 1.39 TB, so 21.7 GB/GPU at 64 GPUs and 2.4 GB at 576. The gather is 0.83 s on the
worst fabric against a **54.7 s** step: 1.5%. The run is compute-bound by a wide margin, and
at N <= 72 the whole job fits inside a single NVLink domain anyway.

PP would add a bubble and, with Block AttnRes, a cache that grows as `pp x vp / cp`. It buys
nothing there is a shortage of.

### Multimodal GRPO / PPO: **negative**

Same memory picture (the reference policy shares the frozen base via `disable_adapter`), plus
the structural problem: RL wants small frequent updates, so `m` is small by design and the
bubble `(p-1)/(v*m + p-1)` has nothing to hide behind. Forward-only logprob passes also make
the AttnRes cache's worst case, since they hold cache no backward will consume.

### Full-parameter post-training: **real but moderate, and it scales with how bad your fabric is**

167 GB of gather against a 9.1 s step is 18% on IB and 37% on EFA. Overlappable on IB;
uncomfortable on EFA, especially since EP's all-to-all competes for the same links. PP
divides the gather by `p`: `pp=2` takes it to 9% / 18%, `pp=8` to 2% / 5%.

So PP is insurance against a narrow fabric, bought with bubble and AttnRes cache.

## Answer 2: without PP, what does the topology have to satisfy?

One inequality. The gather must overlap under the compute:

    dense_gather_bytes / per_GPU_cross_domain_bandwidth  <  compute_time_per_step

Which gives a **bandwidth floor per GPU**, not a topology restriction:

| | floor to overlap at all | with 3x margin |
|---|---|---|
| full parameter | 167 GB / 9.1 s = **18 GB/s** | ~55 GB/s |
| QLoRA | 42 GB / 54.7 s = **0.8 GB/s** | ~2.5 GB/s |

* **QLoRA and GRPO have essentially no requirement.** 0.8 GB/s per GPU is below any ML
  fabric in service. Any cluster works, including RoCE and EFA, and the NVLink domain size
  is irrelevant because the job fits in one domain.
* **Full parameter needs ~18 GB/s and wants ~55.** IB at 100 GB/s has 5.6x margin; EFA at
  50 GB/s has 2.8x, which is enough for the gather alone but thin once EP's all-to-all is
  sharing the wire.

**So there is no natural topology limit for QLoRA or GRPO, and for full parameter the limit
is margin rather than feasibility.** What consumes the margin is expert all-to-all, not the
weight gather -- which is why "put EP inside the NVLink domain" matters more than any PP
decision.

## Answer 3: does PP widen the range of clusters we can serve?

**Yes for bandwidth, no for domain size, and the distinction is the whole answer.**

* **Bandwidth: yes.** PP divides the gather by `p`, so it lowers the per-GPU bandwidth floor
  by the same factor. A fabric at 18 GB/s that cannot run full parameter at `pp=1` runs
  comfortably at `pp=4`. This is real and it is the honest argument for shipping PP.
* **Domain size: only down to ~72.** Making the FSDP mesh fit in a domain needs
  `N/p <= domain`. At N=576 with a 72-GPU domain that is `p=8` -- achievable. With an 8-GPU
  domain it is `p=72`, which 93 layers cannot host and whose bubble would dominate. So PP
  rescues NVL72-class clusters; it does not turn an 8-GPU-domain cluster into a
  large-domain one.

And a caveat that binds the two: at `pp>=4` our AttnRes cache needs offloading, which needs
coherent host memory (0.33 s over NVLink-C2C versus 6.0 s over PCIe for 149 GB). The
hardware with the large domain is also the hardware with the offload path, so the two
requirements coincide -- but on a PCIe-attached accelerator, deep PP is unavailable for this
architecture regardless of the fabric.

## The two customer profiles

### 1M context, pure text

CP is the workhorse; `cp` can go to 96 before the head count binds. At `pp=1` the AttnRes
cache is ~1 GB, so AttnRes is effectively free.

* QLoRA / GRPO: **no PP.** Compute-bound, fits one domain.
* Full parameter: **`pp=1`, or `pp=2` if the fabric is at or below 50 GB/s per GPU.**

### 200K context, heavy video and image-text

The text side is easy (`cp=8` gives 25k tokens local). The pressure is the vision side, and
it does not want PP:

* **`dynamic CP` is what parallelises the tower**, and it rides the CP group -- no TP, no PP
  required. This is the axis to lean on.
* **DEP now runs with LoRA** (fixed 2026-08-12: the vision stage owns no adapted
  parameters, which core treated as an error in three places rather than as a frozen
  stage), but its exchange-buffer defaults hold 8192 patches where one full-resolution
  image is 262144.
* If the tower is frozen, **precomputing its features removes the vision cost entirely**,
  which beats anything PP or DEP can do for it.

So: **no PP**, CP doing double duty over text sequence and vision patches, and feature
caching if the tower is frozen.

## Summary

| scenario | PP value | why |
|---|---|---|
| QLoRA SFT, 1M text | **none** | 1.5% gather against compute, fits one domain |
| QLoRA SFT, 200K multimodal | **none** | same; dynamic CP carries the tower, and DEP buys ~0.6% of a step |
| Multimodal GRPO / PPO | **negative** | small `m` by design; forward-only passes stress the cache |
| Full parameter, IB-class fabric | **low** (`pp=1..2`) | 18% gather, overlaps |
| Full parameter, EFA/RoCE-class | **moderate** (`pp=2..4`) | 37% gather, and EP competes |
| Full parameter, narrow fabric (<20 GB/s) | **high** | PP is what makes it feasible |

**The single sentence:** for the scenarios asked about -- QLoRA SFT and multimodal RL on 2.8T
-- PP is worth nothing, because EP already removes 97% of the gather and those workloads are
compute-bound with state that fits one domain. PP earns its place only in full-parameter
post-training, and its value there is proportional to how narrow the fabric is.

## Addendum, 2026-08-12: I was measuring the wrong term

The section above treats the FSDP weight gather as the fabric's load. It is not the
binding term, and quantifying the one I kept asserting mattered changes the answer to
"does PP widen DC coverage".

### The weight gather is nowhere near binding

At each DC type's memory-driven minimum GPU count, the step is 48-143 s and the gather needs
1-3 GB/s. Turning it around, the scale ceiling at `pp=1` -- the most GPUs whose step is still
long enough to hide 166 GB -- is **6,800 to 22,600 GPUs** depending on the fabric. No
realistic deployment is near it, so the gather gates nothing.

### Expert all-to-all is 5.6x larger, and it does not respond to PP

K3 routes `top_k = 16` of 896 experts, so per token per MoE layer the dispatch+combine volume
is `16 x 7168 x 2 bytes x 2` = **459 KB**. Per rank per step at N=576:

| term | volume | sustained | responds to pp? |
|---|---|---|---|
| **EP all-to-all** | **923 GB** | **28.3 GB/s** | **no** |
| weight all-gather | 166 GB | 5.1 GB/s | divided by `p` |

The invariance is arithmetic, not accident. Per-rank EP volume is
`tokens/(cp*dp) x (92/pp)`, and `cp*dp = N/pp`, so the `pp` cancels:
`tokens x 92 / N`. **Only adding GPUs reduces it.** PP divides layers per rank and multiplies
tokens per rank by the same factor.

### So the answer to "does PP extend DC coverage" is: for full parameter, essentially no

What decides whether a DC can run this is whether **EP fits inside the NVLink domain**. If it
does, that 28 GB/s is intra-node and never touches the fabric, leaving ~5 GB/s of gather --
which every fabric in service satisfies. If EP has to span nodes, 28 GB/s sustained rules out
200G-class RoCE (25 GB/s) and leaves little margin at 400G.

EP must divide 896 and fit the domain. On 8-GPU nodes, **EP=8 achieves it** (112 experts per
rank, expert state still sharded over `EP x efsdp = cp*dp`). So the topology requirement is a
placement decision about EP, and PP is orthogonal to it.

**Corrected claim.** Earlier in this file: "PP divides the gather by p, so it lowers the
per-GPU bandwidth floor by the same factor... this is the honest argument for shipping PP."
Directionally true, quantitatively irrelevant -- it lowers a floor that is already 5x below
the binding term. The honest argument for PP in full-parameter training is not bandwidth.
