# DEP, Dynamic CP and MTP against the report (§5.2.3, §3.3, Table 1)

Read after the fact. Two of my earlier conclusions were reached from my own timing
analysis rather than from the report, and one of them is wrong. This supersedes
the recommendation in `VIT_DEP_DESIGN_2026-08-06.md`.

## Dynamic CP -- what I built is one half, and not the load-bearing half

Report §5.2.3, verbatim in substance:

1. A **single large image is partitioned along the patch dimension across multiple
   devices**, and attention is computed by **gathering key-value pairs (gather-KV)
   across CP ranks**.
2. **Each CP group is divided into several sub-CP groups**, and large images are
   distributed across them **load-balanced**, so the communication fraction does
   not grow with scale.

Purpose, in the report's own words: it reduces encoder latency and cross-device
imbalance "**allowing the remaining encoder computation to be hidden in pipeline
bubbles**". So dynamic CP is a PREREQUISITE for the bubble hiding, not an
independent optimisation.

**What is landed (`57c6728ed`):** whole images distributed round-robin across the
full, static CP group, with a fixed-shape differentiable all-gather. That is the
spirit of (2) with exactly one sub-group -- the whole CP group -- and **none of
(1)**.

**Missing:**

* Intra-image patch-dimension partitioning with gather-KV attention. This is a
  different attention path in MoonViT, not a scheduling change.
* Actual sub-CP-group formation. My note that "static DeviceMesh is not a blocker"
  holds only for what I built; sub-groups genuinely need groups, though
  sub-process-groups suffice and a DeviceMesh is not required.

So the honest label for the landed work is **image-level load balancing across the
CP group**, not "dynamic CP".

## DEP -- the report does the hybrid; my Design A was a weaker substitute

Report §5.2.3, verbatim in substance: DEP (from K2.5) **splits ViT and text
training into separate stages** and balances vision forward and backward passes
across PP stages. Then:

> the ViT forward passes of the **first** PP micro-batches are executed
> **synchronously upfront**, the **remaining** forward passes are scheduled **into
> pipeline bubbles**, and the **backward** passes are handled analogously.

**The dependency inversion I identified is real and the report works around it the
same way** -- the first micro-batches' features are needed immediately, so those
encodes happen upfront. My error was the conclusion: I took "the first ones cannot
go in bubbles" and turned it into "so put all of it in a parallel prologue". The
report puts the *first few* upfront and *the rest* in bubbles, which strictly
dominates -- it gets the upfront cost only for the micro-batches that need it, and
hides the remainder instead of merely dividing it.

So the recommendation changes:

* **Superseded:** Design A (parallel prologue for the whole tower). It leaves every
  bubble empty.
* **Target:** the report's hybrid. Upfront ViT forward for the first PP
  micro-batches; remaining forwards injected into bubbles; backward handled
  analogously. Plus DEP proper: ViT and text as separate stages with vision
  forward/backward balanced across PP stages -- which is a cleaner answer to
  "where do the tower weights live" than my "replicate or shard across the PP
  axis", because the ViT stages own them.

What still stands from the earlier analysis, and matters for implementation: the
delivery path. Injecting work into bubbles means issuing sends that are not in the
schedule's action list, and a send that does not pair with a matching recv in the
expected order is how PP deadlocks. The report does not say how they sequence it;
the AttnRes adapter's design note ("PP owns all NCCL") is the constraint to respect
or consciously break.

## MTP -- Table 1 and §3.3 pin two things

* **Number of MTP Layers: 1.** Same as K2. Our implementation builds
  `num_nextn_predict_layers` layers and the flavor sets 1, which matches.
* **§3.3:** "visual and textual tokens are **interleaved within a single next-token
  prediction objective**", native multimodal from the start rather than a grafted
  encoder with a post-hoc alignment stage.

That second point **settles the multimodal-MTP question left open** in
`attn_res_model._compute_mtp_logits`. The objective is next-token prediction over
the interleaved stream, so depth k's target is the token k+1 ahead **in the spliced
sequence** -- not a shift of `input_ids`. The guard that currently raises is
therefore correct to refuse, and the fix is to source targets from the spliced
sequence, not to hand the embedding table back.

Also from Table 1, for the record: ViT is 401M over 27 layers, patch size 14,
**12 attention heads** -- which confirms the 3-head debug tower is a fixture
artifact, and that `kimi_k3_debugmodel_report_arch_vit4h` exists only to make head
sharding testable at all.
