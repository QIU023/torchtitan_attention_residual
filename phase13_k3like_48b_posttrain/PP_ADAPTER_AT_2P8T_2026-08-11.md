# The PP adapter at real 2.8T: mechanism, what the report fixes, and what breaks

Two questions, answered separately because they have different answers.

## 1. Frozen-tower feature caching: which scenario

Only one, and it is narrow: **the tower is frozen AND the image set is reused across
steps.** That is LoRA SFT and full-parameter text-only SFT over a fixed multimodal corpus,
where every epoch sends the same pixels through an unchanging encoder and gets the same
features back. There, the encoder does not belong in the training loop at all -- precompute
`[N_patches, D]` per image once, store it, and the vision tower never runs during training.
That is a 100% removal of the vision cost, against DEP's ~56% hiding of something worth
0.6% of a step.

It does NOT apply to:

* **pretraining** -- the report trains MoonViT-V2 jointly from scratch (sec 2.4), so the
  encoder changes every step and yesterday's features are wrong;
* **any run that adapts the tower**, including LoRA with tower targets;
* **online RL / GRPO** where prompts carry newly generated or sampled images;
* **augmentation** that resizes or crops per epoch -- the cache key is the exact pixel
  input, so augmentation defeats it.

So it is a real win in exactly the setting where DEP is worthless, and worthless in exactly
the setting where DEP is defined. They are not competitors; they address disjoint cases.
The cost is storage: at 8192 patches x 7168 dims x 2 bytes that is 112 MiB per large image,
so a corpus of any size needs a real store rather than RAM.

## 2. How our PP AttnRes cache works

Block Attention Residuals need, at layer L, the block-aggregated output of every EARLIER
block. Under PP those earlier blocks live on other stages, so the mechanism has two halves.

**Cross-stage (the real transport).** Blocks produced on stage S and needed on stage S+1..n
ride the existing PP activation channel: they are concatenated onto the stage output as a
`recv_delta` tensor, so PP's own SEND_F carries them forward and SEND_B carries their
gradients back. Nothing bespoke is sent, and the backward is PP's, not ours -- which is why
the adapter needs no gradient collectives.

**Same-rank across virtual stages (the subtle half).** Under Interleaved1F1B one physical
rank owns several virtual stages, so a block committed by virtual stage v can be read back
by virtual stage v+k on the SAME rank without ever crossing the wire. That path is a
rank-local `RankLocalCache` keyed by microbatch, holding the block tensors autograd-live
against their producers, plus `(mb, producer_stage, block_idx)` metadata.

The autograd hazard there is the part worth knowing: the consumer's backward would traverse
into the producer's forward graph and free it, and the producer's own backward -- driven
later by SEND_B -- would then fail with "backward through the graph a second time". A
process-global `retain_graph=True` fixed it and leaked every unrelated graph; an
`autograd.Function` wrapper did not hold under FSDP plus selective AC. What holds is a
producer-side grad hook plus a consumer-side **detached leaf**: with no upstream `grad_fn`
there is nothing to traverse. Recv-originated blocks are deliberately left attached, so
PP's SEND_B still carries their gradient to the producing rank.

Eviction is a step-end sweep over `set(_seen_mbs) | set(_blocks)`, which matters because a
FORWARD-ONLY microbatch never marks itself via backward and would otherwise never leave.

## 3. What the report specifies for PP and DEP

**It publishes no numbers.** Section 5.2.3 states three things definitionally -- ViT and
text are separate stages, vision forward and backward are balanced ACROSS stages (so more
than one ViT stage), and the first micro-batches' ViT forwards run upfront with the rest
scheduled into bubbles. It gives no pipeline degree, no virtual-stage count, no ViT stage
count, and no per-stage layer split. Anyone quoting "the report's PP settings" is quoting
something that is not there.

What that means for us: `n_vit` and `pp_degree` are OURS to choose, and the only
report-derived constraint is `n_vit >= 2` if DEP is to mean what the report says. Our
implementation supports `n_vit > 1` and it is verified exact from a shared checkpoint, but
every DEP measurement here has used `n_vit = 1`, which satisfies requirement 1 and not
requirement 2.

## 4. Scaling this adapter to real 2.8T: the concrete problems

`kimi_k3_2p8t_block_attn_res`: 93 layers, hidden 7168, **8 AttnRes blocks** of 12 layers
(7 full + a 9-layer tail), 896 experts, seq_len 8192.

### 4.1 The rank-local cache is the dominant new memory term, and it scales with in-flight microbatches

One cached block is `[seq, hidden]` in bf16. At 2.8T dims:

| seq | one block | x8 blocks = one microbatch |
|---|---|---|
| 4096 | 0.055 GiB | **0.44 GiB** |
| 8192 | 0.109 GiB | **0.88 GiB** |
| 131072 | 1.75 GiB | **14.0 GiB** |

The cache holds every LIVE microbatch's blocks until that microbatch's backward completes,
and under Interleaved1F1B the number in flight on a rank is on the order of
`pp_degree x vp_degree`. At pp8 x vp4 and seq 8192 that is up to ~28 GiB per rank of pure
AttnRes cache, on top of weights, optimizer state and ordinary activations. At the 1M-token
context the report targets, a single microbatch's blocks exceed any device.

This is the first thing to measure at scale, and it is not a defect -- it is the design's
cost, and it was invisible at the debug flavor where hidden is 256 and one block is 4 MiB.

**Mitigations, none implemented:** keep the cache in bf16 regardless of `training.dtype`
(it is already bf16 in mixed precision, but an fp32 run doubles it); store only the blocks a
LATER virtual stage on this rank actually reads, which the layout already knows;
recompute rather than cache when the producer is on the same rank and cheap.

### 4.2 The DEP exchange buffer's defaults are debug-sized by a factor of 32

The ViT-to-text hop sends a FIXED-shape patch stream, sized
`dep_max_images x dep_max_grid_h x dep_max_grid_w` = 8 x 32 x 32 = **8192 patches**. It has
to be fixed because PP sizes its P2P buffers once, and a batch that exceeds it raises at the
sender rather than truncating -- which is the right behaviour.

But `navit_resize` permits 512 patches per side, so ONE real image can be 512 x 512 =
262144 patches, **32x the entire default buffer**. Any real multimodal run needs these
raised, and raising them costs fixed P2P bandwidth on every hop whether or not the batch
uses it. Sizing them is a real decision at scale, and today it is three defaults nobody has
tuned against a real corpus.

### 4.3 The 93-layer stack does not divide evenly, and the tail block is special

8 blocks x 12 = 96 > 93, so the partition is 7 full blocks plus a 9-layer tail. That
asymmetry has already produced one bug class here (finding 41: block size reaching the model
only through a lossy `num_blocks` round trip, which changed k3mini's partition from 11+10 to
12+9). At 2.8T the tail block is 25% smaller than the others, so per-stage cost is uneven
and the PP schedule's balance assumption is slightly wrong. Nothing is incorrect; the
pipeline is just not as balanced as the layer count suggests.

### 4.4 GRPO adds forward-only passes, which is the cache's worst case

An RL step runs the actor forward+backward, but also reference-policy and old-policy
logprob passes that are FORWARD ONLY. Those cache blocks and never mark a backward, so they
survive until the step-end sweep -- exactly the leak finding 65 fixed. At 2.8T that means a
logprob pass over a long context transiently holds 0.88 GiB (seq 8192) to 14 GiB (128k) per
microbatch of cache that no backward will consume.

The sweep handles correctness. What is untested is the PEAK: a GRPO step's high-water mark
includes both the actor's live cache and the forward-only passes' dead cache, and no run
here has measured that at any scale. **This is the single most likely thing to fail first
when scaling GRPO**, and it is measurable today at 48B rather than 2.8T.

### 4.5 n_vit > 1 has never run at depth

The split tower is verified exact from a shared checkpoint at 13 and 30 layers, and it only
became runnable under FSDP after a fix this week (the shares were calling
`forward_head`/`forward_body`/`forward_tail` directly, bypassing FSDP2's all-gather, so the
split tower had never run under FSDP at all). At 93 layers with a 447M-parameter tower split
across stages, nothing has exercised it. The report REQUIRES more than one ViT stage, so
this is on the path to a faithful DEP, not optional.

### 4.6 EP at 896 experts is orthogonal to the cache but not to the schedule

The cached blocks are activations, so EP does not multiply them. But EP introduces
device-to-host syncs that upstream's own comment says interfere with FSDP's implicit
prefetching -- which is why `apply_fsdp_to_decoder` sets up explicit prefetch when
`ep_degree > 1`. Our adapter's same-rank cache reads happen inside that same schedule, and
their interaction with EP prefetching at 896 experts is unmeasured.

## 5. What I would do first

1. **Measure the GRPO peak at 48B**, not 2.8T. 4.4 is the most likely first failure and
   the cheapest to see.
2. **Instrument the cache high-water mark** per rank per step, and put it in the metrics.
   It is the term nobody is watching and it grows with `pp x vp x blocks x seq x hidden`.
3. **Size the DEP buffers against a real corpus** before claiming DEP works at scale; the
   current defaults cannot hold one full-resolution image.
4. **Run `n_vit = 2` on the 30-layer flavor under FSDP+PP** as a rehearsal for depth, since
   that path is newly working and untested beyond exactness.
5. Only then argue about DEP's benefit, which on this hardware is arithmetic anyway.
