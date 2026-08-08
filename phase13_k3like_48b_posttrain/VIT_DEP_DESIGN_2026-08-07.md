# ViT DEP, designed against the report and the schedule API

Supersedes `VIT_DEP_DESIGN_2026-08-06.md` entirely. That one proposed a parallel
prologue derived from my own timing analysis; the report prescribes something
different and strictly better, and the schedule API turns out to permit it.

## What the report asks for, verbatim (5.2.3)

> In Kimi K2.5, we introduced the Decoupled Encoder Process (DEP), which splits
> ViT and text training into separate stages and balances vision forward and
> backward passes across PP stages. We observe that, under the interleaved 1F1B
> pipeline schedule, the text forward passes of the first PP micro-batches are all
> scheduled at the very beginning, while the text backward passes of the last PP
> micro-batches finish only at the very end. We therefore further decompose the ViT
> computation. The ViT forward passes of the first PP micro-batches are executed
> synchronously upfront, the remaining forward passes are scheduled into pipeline
> bubbles, and the backward passes are handled analogously.

**Three requirements, all definitional.** An earlier version of this document
demoted the second to "step 3, if the measured share justifies it". That was wrong:
the report puts it in the sentence that says what DEP IS.

1. ViT and text are **separate stages**.
2. Vision forward and backward passes are **balanced across PP stages** -- so the
   ViT occupies more than one stage, by definition, not as a later optimisation.
3. The **first** micro-batches' ViT forwards run synchronously upfront; the
   **remaining** ones go into bubbles; backward analogously.

(3) is what falls out of (1) and (2) under an interleaved schedule rather than a
separate mechanism -- see the API note below. Figure 11 corroborates the shape:
``ViT fwd`` sits at the far left of the PP timeline and ``ViT fwd`` / ``reduce
grad`` at the far right, with the 1F1B body between them.

**What the report does NOT specify, and cannot:** how to express any of this in
torchtitan. The report describes Kimi's own infrastructure (MoonEP and their own PP
implementation). Everything below under "Sequence" and the two splitter options are
MY engineering choices for this codebase, and they are labelled as such. Do not read
them as report-derived.

## The API fact that changes the design

`torch.distributed.pipelining.schedules._PipelineScheduleRuntime` exposes
`register_custom_function(computation_type, fn)`, which looked like the hook for
injecting ViT work into bubbles. It is not, and the reason is a closed set: the
call validates `computation_type` against exactly `FORWARD`, `FULL_BACKWARD`,
`BACKWARD_INPUT`, `BACKWARD_WEIGHT`, `OVERLAP_F_B`, `UNSHARD`, `RESHARD`,
`REDUCE_GRAD`. **A new action type such as `VIT_FORWARD` cannot be added**, so
"inject an extra action into the bubble" is not expressible.

That is a better outcome than it sounds, because the alternative I had recorded as
the risk -- issuing sends outside the schedule's action list, which is how PP
deadlocks and which would give up the property the AttnRes adapter is built around
("PP owns all NCCL") -- is not needed either. **If the ViT is a stage, its
FORWARD and BACKWARD actions are already in the action list**, and the interleaved
schedule places them where bubbles would otherwise be. The report's phrasing is
not describing an injection mechanism; it is describing what the schedule does
once the encoder is decoupled.

So the pipeline stays linear: ViT stage -> text stages. No reverse-direction
channel, no out-of-band send, no new action type.

## What stands in the way, concretely (my analysis of this codebase, not the report)

The current adapter takes the opposite position on purpose. From
`pipeline_adapter._unwrap_multimodal_for_pp`:

> Vision features are spliced into the embeddings, so the tower belongs with
> whichever chunk kept `embed_tokens` -- nothing vision-side crosses a stage
> boundary.

Four things follow from moving it, and each is a real change rather than a rename:

1. **Stage count.** Today the split is over text layers only. DEP adds ViT stages
   ahead of them, so `pipeline_parallel_degree` no longer equals the number of text
   chunks. Requirement (2) makes it more than one ViT stage, not "may be".

2. **Stage 0's input is not a hidden state.** It is `pixel_values` plus
   `grid_thw`. `pipeline_llm` assumes a homogeneous chain whose stages exchange one
   activation tensor.

3. **The text stage needs BOTH.** The splice replaces vision sentinels in the token
   stream, so the first text stage needs the ViT features from the previous stage
   AND `input_ids`. Per-stage batch inputs exist in torchtitan, but the splice
   currently lives inside the multimodal wrapper's forward, which owns both.

4. **Variable feature length.** A batch's images produce a variable number of
   tokens, and PP's P2P wants fixed shapes. Dynamic CP already forced this problem
   once and the answer there works here: `grid_thw` is replicated, so every stage
   can size the exchange from it without communicating. `merged_tokens(h, w, kh,
   kw)` is the per-image length, and it carries no `t` -- the projector collapses
   time.

## Sequence (my plan, not the report's)

1. **Make the ViT a stage with a fixed-shape output contract.** Feature length
   from `grid_thw` via `merged_tokens`, padded to the batch's maximum. Verify
   against the non-PP path: distributing the tower changes no arithmetic, so
   step-1 loss and grad_norm should match to the bf16 floor, and the floor is real
   -- see below.
2. **Move the splice to the first text stage**, taking features as the incoming
   activation and `input_ids` from the batch.
3. **Balance the vision passes across more than one ViT stage.** Report-required,
   not conditional -- see requirement (2). What IS open to measurement is how many
   stages and how the split is chosen, since the report gives no rule.
4. **Measure the bubble occupancy**, not just correctness. The claim to make is
   "most of the ViT computation is hidden", and that needs a profile, not a loss.

## The measurement floor, which is not optional to know

Any A/B involving the tower has a **bf16 floor**, established while landing dynamic
CP: tensors arriving at `MoonViT.forward` are bf16 even under
`--training.dtype float32`, because the `"fsdp"` mesh here is `dp_shard x cp`, so
CP alone puts FSDP in the path and FSDP all-gathers the fp32 master into a bf16
compute copy. Reading `patch_embed.proj.weight` before the forward hook shows fp32
and is misleading. The observed per-feature delta under a correct partition was
1.56e-02 against a magnitude of 4.278 -- 3.6e-03 relative, exactly `2**-8`.

So the tower's arithmetic must be verified in a standalone fp32 reproducer, where
nothing casts, and the training A/B judged against the bf16 floor. Dynamic CP is
verified that way: `max|delta| = 0.000e+00` at two ranks on four grids including
the report-arch config's own and a frame-spanning video.

## The adapter's approach is the report's, which is worth knowing

Section 5.2.2, on memory-efficient training, describes exactly what our cross-stage
adapter does:

> For pipeline parallelism, we adopt cache-based pipeline communication [57], in
> which only newly generated blocks are incrementally transferred between stages and
> released as soon as the micro-batch finishes, reaching the theoretical lower bound
> on memory footprint.

Reference [57] is the AttnRes paper. So transferring only the delta of newly
committed blocks is K3's own choice, not an invention of ours that happens to work.

## What is NOT in the way

The AttnRes cross-stage adapter. It changes the CONTENT of what crosses a PP hop
(the delta of committed blocks instead of the full stack) and adds no collectives,
so it is orthogonal to where the tower runs. Verified independently: multimodal
PP8xVP4 is bit-identical with the adapter on and off, seq 1024 through 8192.

---

# Implementation route, verified against core (2026-08-08)

The two options offered earlier -- hoist the text stack's children, or build the ViT
stage manually -- are both unnecessary. Reading core rather than reasoning about it
turned up a route that needs **no core change and no parameter-namespace change**.
Three facts, each read off the source:

**1. `parallelize_fn`'s return value replaces the stage's module.**
`pipeline_parallel.py` builds every stage from `_split_module`'s chunk and then does:

```python
m = parallelize_fn(m, ...)
model_parts[i] = m
stages[i].submod = m        # "update the model in the stage"
```

So a chunk can be swapped for a completely different module after the
`PipelineStage` exists. The current adapter already relies on this to attach the
tower to the embed-owning chunk. Therefore: give
`module_fqns_per_model_part` one entry that matches nothing in the text model
(`_split_module` sets every non-matching child to None, yielding a zero-parameter
chunk), and return a purpose-built ViT stage module for it. `language_model.*`
naming is untouched, so the checkpoint contract, `hf_key_map` and
`state_dict_adapter` are all unaffected.

**2. Kwargs reach EVERY stage; positional args only the first.** From
`trainer.py`:

```python
self.pp_schedule.step(
    arg_mbs=arg_mbs if self.pp_has_first_stage else None,
    kwarg_mbs=kwarg_mbs,      # NOT gated on pp_has_first_stage
    ...
)
```

**3. Stages can exchange a tuple**, so the ViT stage returns
`(features, input_ids)` and the first text stage receives both positionally. That
avoids needing `input_ids` in `extra_kwargs`, which would have been a core change.
Both elements have fixed shapes: `input_ids` is `[B, T]`, and `features` is the
config-derived capacity buffer from `stage_exchange_capacity` -- which is why that
capacity must not be batch-derived (PP sizes its P2P once).

## The resulting stage layout

```
stage 0        ViT stage(s)   in: input_ids (arg) + pixel_values, grid_thw (kwargs)
                              out: (packed features, input_ids)
stage 1        embed_tokens + first text layers
                              in: (packed features, input_ids)
                              splices, then runs its layers
stage 2..P-1   text layers as today
```

Requirement (2) -- vision forward and backward balanced across PP stages -- means
more than one ViT stage, i.e. more than one such no-match FQN entry, with the tower's
blocks distributed between them. The report gives no rule for how many, so that is
the part to choose by measurement.

## Status

* **Landed:** the exchange contract (`stage_exchange_lengths`,
  `stage_exchange_capacity`, `pack_stage_features`, `unpack_stage_features`) with
  tests, including that lengths carry no frame count and that overflow raises rather
  than truncating.
* **Not yet written:** the ViT stage module, the no-match FQN entry, the splice moved
  into the first text stage, and the split of the tower across several ViT stages.
* **First verification when it exists:** at pp2 on the multimodal debug flavor,
  step-1 loss and grad_norm against the non-DEP path. Distributing the tower changes
  no arithmetic, so the target is equality to the bf16 floor -- and the floor is
  real, see the section above. Then a profile for the bubble-occupancy claim, because
  "most of the ViT computation is hidden" is not a loss statement.
