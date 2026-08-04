# Scope note: this PR is CP correctness, not the report's dynamic CP

Refs: pytorch/torchtitan#3029

Keep the distinction in the PR wording. What we ship is the multimodal CP
*correctness* baseline that no upstream model currently has. The report's
sec 5.2.3 is a performance layer on top of it, and claiming the second while
shipping the first would not survive review.

## What the report actually describes (sec 5.2.3, verbatim)

> A single large image is partitioned along the patch dimension across multiple
> devices, and attention is computed by gathering key-value pairs (gather-KV)
> across CP ranks. In addition, we divide each CP group into several sub-CP
> groups and distribute multiple large images across them in a load-balanced
> manner, preventing the communication fraction from growing with scale.

Three separable pieces, plus a fourth in the same section:

1. Patch-dimension sharding of a single large image across devices.
2. gather-KV attention across CP ranks (not ring).
3. Sub-CP groups with load-balanced assignment of multiple large images, so
   the communication fraction does not grow with CP degree.
4. "Encoder computation in PP bubbles": the DEP line from K2.5 [59], extended
   by scheduling most ViT forward and backward into interleaved-1F1B bubbles.

Note the dependency the report states: (1)-(3) reduce encoder latency and
imbalance, "allowing the remaining encoder computation to be hidden in
pipeline bubbles". So (4) rides on (1)-(3); doing DEP alone does not reproduce
the reported effect.

"Dynamic" is not separately defined in the report. Two things in the text are
data-dependent -- it is applied to *large* samples specifically, and the sub-CP
grouping follows the batch's image distribution -- so the grouping varies per
batch rather than being a fixed degree applied uniformly. That reading is ours,
not the report's wording.

## What this PR does instead

Every CP rank encodes every image redundantly, then selects its own slice of
the encoder output by a prefix sum over per-rank sentinel counts. No FLOP
saving, no communication saving, no load balancing. It makes CP numerically
correct for the multimodal path, which is the part upstream does not have --
`qwen3_5`, `kimi_k2_7` and `kimi_k3` all refuse CP with `NotImplementedError`,
and two name the vision scatter as the reason.

The text-side sequence CP this PR carries (Ulysses for MLA, fla's KCP for KDA)
is required either way; dynamic CP does not replace it.

## Feasibility if we do pursue it, in difficulty order

* **Patch-dim sharding + gather-KV: easiest.** MoonViT is varlen with
  `is_causal=False`, so gathering K/V needs none of ring attention's causal
  block bookkeeping.
* **Sub-CP groups: hardest, and the least torchtitan-native.** `DeviceMesh` is
  static for the run, while the report regroups per batch. A native version
  would have to pre-create every candidate sub-group and select one per batch,
  which is exactly where "dynamic" collides with the framework's assumption.
  This is a core change, not a model-folder one.
* **Encoder in PP bubbles: needs a new extension point** in the pipeline
  schedule to inject ViT forward/backward into 1F1B bubbles. No such hook
  today.

## Wording to use in the PR

Say "multimodal context parallelism (correctness); the encoder still runs
redundantly per CP rank" and cite sec 5.2.3 as explicitly out of scope with
the three-part breakdown above. Do not say "dynamic CP".
