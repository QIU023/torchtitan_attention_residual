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

---

# Requirement (2) is ambiguous in the report, and here is how I read it

"balances vision forward and backward passes across PP stages" admits two readings,
and the report does not disambiguate them. Recording both, because picking one
silently would be exactly the kind of invention this project keeps correcting.

**Reading A -- split the tower's DEPTH.** The ViT's encoder blocks are distributed
over several vision stages the way text layers are over text stages. Pipeline-native
and needs nothing the linear pipeline does not already have.

**Reading B -- distribute the per-micro-batch PASSES.** The word "passes", and the
next sentence's "The ViT forward passes of the first PP micro-batches ... the
remaining forward passes are scheduled into pipeline bubbles", both point at
per-micro-batch work rather than at layers. Taken alone this needs the tower
replicated on every stage and its output delivered AGAINST the pipeline direction,
which the report never describes.

**Figure 11 does not settle it.** The ``ViT fwd`` boxes sit in the top annotation
row that labels what overlaps in each phase -- one at the far left, one (green) at
the far right beside ``reduce grad`` -- and NOT spread across the per-rank rows,
which is what would have shown reading B.

**How they reconcile, which is what I implement.** Under a schedule with virtual
stages a single rank holds several stages. So if there are ``n_vit`` vision stages
distributed across ranks, each rank owns vision work AND text work, and the
interleaved schedule places that rank's vision actions in its own text bubbles.
That is reading A's mechanism producing reading B's effect, with no
reverse-direction delivery and no replicated tower. It is also the only one of the
two that the closed set of computation types permits.

I am not claiming the report says A. I am claiming A achieves what the report
describes, and that B as literally stated needs a delivery path the report does not
give.

## What Reading A still needs

The intermediate exchange between two vision stages carries the ViT's HIDDEN patch
stream, whose length is the batch's patch count and therefore varies. Same
constraint as the feature exchange and the same answer: a config-level patch
capacity, with the real length recomputed on the receiving stage from ``grid_thw``,
which is replicated and reaches every stage as a kwarg. Under dynamic CP the local
patch count varies per rank too, so the capacity is per-rank-local and derived from
the same maxima.

Stage layout for ``n_vit > 1``:

```
vision 0        patch_embed + encoder blocks [0, k)        -> padded patch stream
vision i        encoder blocks [.., ..)                    -> padded patch stream
vision n-1      blocks + final_layernorm + mm_projector
                + embed_tokens + splice                    -> spliced embeddings
text 0..        transformer layers
```

``embed_tokens`` and the splice stay on the LAST vision stage for the reason
established at ``n_vit = 1``: ids cannot cross the pipe, because PP's metadata
inference pushes dummy values through it and indexing an embedding with those
asserts.

## Status at ``n_vit = 1``

Implemented and verified at two pipeline degrees, from a shared seed checkpoint.

| | loss (step 1) | grad_norm |
|---|---|---|
| single GPU | 12.04902 | 15.1250 |
| pp2 non-DEP | **12.05471** | 15.3125 |
| pp2 DEP | **12.05471** | 15.2500 |
| pp4 non-DEP | **12.07418** | **12.5000** |
| pp4 DEP | **12.07418** | **12.5000** |

At pp2 the step-1 loss is identical to five decimals and grad_norm differs by one
bf16 step (both are multiples of 1/16). At pp4 **both** are identical. So moving the
tower onto its own stage changes no forward arithmetic, and the mechanism does not
depend on the stage count. Both PP legs sit 5.69e-03 from single-GPU at pp2, non-DEP
included, so that is PP microbatching rather than DEP.

**Cold start is not a valid comparison here** and gave a misleading 0.05: different
stage splits consume RNG in a different order, so the weights differ. The seeded
protocol is not optional for anything that changes the split.

**A stack fact that cost three runs:** ``n_microbatches`` equals
``local_batch_size``, not ``global / local`` -- since #3856 the dataloader emits one
pipeline microbatch per fetch. So pp4 needs ``local_batch_size >= 4``, and
``local_batch_size 1`` yields one microbatch regardless of the global batch. Each of
those three failures took the non-DEP leg down with it, which is the signal that the
fault was in the harness and not in what was being measured.

## The blocker for ``n_vit > 1``, located precisely

With one vision stage, the splice sits on it and reads ``input_ids`` from the
positional argument, which torchtitan gives to the FIRST stage. With more than one,
the splice belongs on the LAST vision stage -- the features do not exist before it --
and that stage is not the first, so it has no ids. And ids cannot cross the pipe:
PP's metadata inference pushes dummy values through, and indexing an embedding with
those asserts (measured).

Passing the text EMBEDDINGS plus a float sentinel mask instead would keep the pipe
all-float, but the splice then needs the mask's True count to match the feature
count, and during metadata inference it will not -- so the splice would raise unless
it clamps, and clamping hides real mismatches rather than reporting them.

**So the ids have to arrive as a kwarg, which reach every stage** (``trainer.py``
passes ``kwarg_mbs`` unconditionally while gating ``arg_mbs`` on
``pp_has_first_stage``). They come from the dataloader's ``input_dict``, and
everything except the ``"input"`` key becomes a forward kwarg. The obstacle is that
the multimodal collator is **hardcoded**: ``hf_datasets/multimodal/mm_datasets.py``
constructs ``MultiModalCollator(...)`` inline at its dataloader build, with no
config field to substitute one. Editing that is a core change, which experiments do
not get to make.

The way through, entirely inside the model folder: a ``Kimi`` dataloader config that
builds the core ``MMDataLoader`` and wraps its iterator to add ``text_ids`` to each
batch dict. That is configuring an existing extension surface rather than modifying
core, the same shape as everything else in this folder. Once ``text_ids`` exists,
the splice can live on the last vision stage -- or move back to the first TEXT
stage, which is arguably closer to "ViT and text as separate stages" than putting
``embed_tokens`` on the vision side.

## Capacity must agree with the dataloader, and currently does only by coincidence

The multimodal flavor configures ``max_images_per_batch=8``, ``max_patches=1024``
and ``max_patches_per_side=64``. The DEP defaults are ``dep_max_images=8`` and a
32x32 grid, giving ``8 * 16 * 16 = 2048`` merged tokens -- consistent with 8 images
of at most 1024 patches each. Nothing enforces the agreement, though.
``pack_stage_features`` raising on overflow is what makes a mismatch loud instead of
silent, which is the property that matters; deriving the DEP maxima from the
dataloader config would be better and is not done.

## Interleaved1F1B: the configuration the bubble claim needs

Under plain 1F1B at ``n_vit = 1`` and pp2 there are two stages: rank 0 holds only the
vision stage and rank 1 all the text layers. **Rank 0 has no bubbles to fill** -- it
spends the step encoding. That configuration can show the arithmetic is unchanged,
and does, but it cannot show anything about hiding work.

With virtual stages a rank holds several, and that is where the report's mechanism
lives. Measured, pp2 + Interleaved1F1B + DEP:

```
stage 0 (rank 0)  ['__kimi_dep_vision__0', 'embed_tokens']
stage 1 (rank 1)  layers.0-3
stage 2 (rank 0)  layers.4-7          <- same rank as the vision stage
stage 3 (rank 1)  layers.9-12, norm, lm_head
```

Rank 0 owns the vision stage AND a text stage, so its vision actions sit in its own
text pipeline's gaps. That is reading A's mechanism producing reading B's effect,
which is what this design claimed and had not yet shown.

``loss 12.07418`` and ``grad_norm 12.50``, **both identical to non-DEP** under the
same schedule.

## Correctness of ``n_vit = 1``, complete

| configuration | loss | grad_norm |
|---|---|---|
| pp2, 1F1B | identical to non-DEP (12.05471) | 1 bf16 step apart |
| pp4, 1F1B | **identical** (12.07418) | **identical** (12.5000) |
| pp2, Interleaved1F1B | **identical** (12.07418) | **identical** (12.50) |

What remains for the report's claim is a PROFILE. "Most of the ViT computation is
hidden within pipeline bubbles" is a statement about occupancy, and none of the above
measures it -- equal losses would hold whether the vision work overlapped perfectly
or not at all.

---

# NEGATIVE RESULT (2026-08-08): ViT-as-a-stage does not fill bubbles

This refutes a claim made earlier in this document. It said:

> If the ViT is a stage, its FORWARD and BACKWARD actions are already in the action
> list, and the interleaved schedule places them where bubbles would otherwise be.

That was an assumption. It is wrong, and the schedule's own structure shows it.

## The measurement

Not wall clock -- this box is 8x RTX 5060 Ti over PCIe and its communication
fraction does not extrapolate to the report's fabric. Instead, torch's dependency
simulator (``_simulate_comms_compute`` over ``pipeline_order_with_comms``) gives a
per-rank slot list in which ``None`` is a bubble: a slot where the rank could execute
nothing because a dependency was unmet. Probe:
``dep_bubble_structure.py``. FSDP's UNSHARD/RESHARD/REDUCE_GRAD actions are filtered
out first; they are parameter management, not pipeline dependencies, and the
simulator raises on them.

pp8, Interleaved1F1B, 16 virtual stages, 16 microbatches, 30-layer flavor:

| rank | non-DEP slots / bubbles | DEP slots / bubbles | DEP vision slots | stages held |
|---|---|---|---|---|
| 0 | 443 / **283** | 443 / **283** | **64** | [0, 8] |
| 1 | 442 / 250 | 442 / 250 | 0 | [1, 9] |
| 2 | 440 / 248 | 440 / 248 | 0 | [2, 10] |
| 3 | 438 / 246 | 438 / 246 | 0 | [3, 11] |
| 4 | 436 / 244 | 436 / 244 | 0 | [4, 12] |
| 5 | 434 / 242 | 434 / 242 | 0 | [5, 13] |
| 6 | 432 / 240 | 432 / 240 | 0 | [6, 14] |
| 7 | 426 / 266 | 426 / 266 | 0 | [7, 15] |
| total | **2019 bubbles** | **2019 bubbles** | | |

**The bubble count is identical.** Not smaller, not larger.

## Why, and it is structural rather than a bug

A pipeline stage's actions sit on the critical path. The schedule does not know one
of its stages is a vision encoder; it places stage 0's forwards and backwards exactly
where stage 0's work goes. And because the vision stage is taken OUT of the text
budget, DEP swaps a text stage for a vision stage: same number of stages, same
dependency graph, same stalls. The vision work occupies a COMPUTE slot that a text
stage would have occupied. It does not occupy a bubble.

"Hidden within pipeline bubbles" requires the opposite: keep every text stage AND
place vision work into the idle slots. Those are different operations, and making the
ViT a stage performs the first one.

The same table also quantifies the missing requirement (2): 64 vision slots on rank 0
and **zero on the other seven ranks**. "Balances vision forward and backward passes
across PP stages" is not partially satisfied, it is not satisfied at all -- and that
is now counted rather than inferred.

## What this leaves standing, and what it does not

Standing: requirement (1). ViT and text ARE separate stages, and it is numerically
exact -- pp2/1F1B, pp4/1F1B and pp2/Interleaved1F1B all reproduce the non-DEP loss,
the last two bit-identical on grad_norm too. That is worth keeping regardless: it is
the decoupling the rest depends on, and it makes the tower's placement a
configuration rather than a hard-coded attachment to the embed stage.

Not standing: any claim about hiding vision compute. Requirement (3) is not
implemented either, because the mechanism I assumed would deliver it does not.

## What the report probably means, stated as a reading rather than a fact

"The ViT forward passes of the first PP micro-batches are executed synchronously
upfront, the remaining forward passes are scheduled into pipeline bubbles" reads, on
this evidence, as software-pipelining the ViT AGAINST the text pipeline on the same
rank -- microbatch j's vision encode running concurrently with microbatch j-k's text
compute, on a side stream -- rather than as inserting the ViT into the pipeline's
dependency chain. Once it is in the chain it is on the critical path by construction.

That sits awkwardly with "splits ViT and text training into separate stages", so
their "stage" may not mean a PP stage. I am not going to resolve the report's wording
by fiat. What I do have is a measurement showing the route I took does not produce the
benefit it claims, which is the part that matters for deciding what to do next.

The closed set of computation types blocks the obvious implementation of the other
reading -- no new action type can be registered -- so delivering it needs either a
side-stream mechanism outside the schedule (and then the "PP owns all NCCL" property
has to be given up deliberately) or an upstream change.

---

# Re-read of 5.2.3 word by word (2026-08-08): two of my readings were wrong

## The decomposition is per MICRO-BATCH, not per depth

"We therefore further decompose the ViT computation. The ViT forward passes of the
**first PP micro-batches** are executed synchronously upfront, the **remaining**
forward passes are scheduled into pipeline bubbles."

The unit is the micro-batch. So "Reading A" earlier in this document -- distributing
the encoder's BLOCKS across several vision stages -- is not what "further decompose
the ViT computation" means, and the ``n_vit > 1`` plan built on it was solving the
wrong problem. That also retires the vision-to-vision hidden-patch-stream exchange
that plan needed.

## The observation names the CONSTRAINT, not the opportunity

"the text forward passes of the first PP micro-batches are all scheduled at the very
beginning, while the text backward passes of the last PP micro-batches finish only at
the very end."

Read as a constraint it is immediate: the first micro-batches' text forwards happen
at once, so their ViT forwards cannot be deferred -- hence "synchronously upfront".
The LATER micro-batches' text forwards happen later, so their ViT forwards have slack
and can be moved into idle slots ahead of them. Symmetrically the last
micro-batches' text backwards finish at the end, so those ViT backwards can go late.

So "balances vision forward and backward passes across PP stages" means distributing
the per-micro-batch vision passes over the stages' idle slots. Which is what the
negative result above already pointed at.

## The mechanism this needs, and it is available

Not "make the ViT a stage and hope the scheduler helps" -- that was measured and it
does not (2019 bubbles either way). It needs the ACTION LIST REORDERED so vision
actions land where a rank would otherwise stall.

That surface exists and is now verified rather than assumed:

* ``_PipelineScheduleRuntime._load_csv(path, format="compute_only")`` takes a
  per-rank COMPUTE action table and re-runs the lowering passes to regenerate the
  comms schedule.
* Round-tripped: dumping ``pipeline_order``, reloading it unchanged and re-simulating
  gives **slots 443, bubbles 2019, identical** -- so replacing the IR is
  behaviour-preserving when the reorder is the identity, which is the precondition
  for trusting a non-identity one.
* torch's own simulator docstring blesses the approach: "the total number of
  simulator steps can be used as a metric for unit tests involving IR optimization
  passes as **reordering and merging of IR can reduce the number of simulated
  steps**."

## The plan, with a hardware-independent pass/fail

1. Build the Interleaved1F1B compute IR as today.
2. Choose ``J``: the number of leading micro-batches whose ViT forwards must run
   upfront. The report does not give a rule; the constraint it states is that a
   micro-batch's ViT forward must precede its text forward, so ``J`` follows from the
   warmup depth of the schedule.
3. Reorder: leave the first ``J`` vision forwards where they are; move the rest to
   positions where the owning rank stalls but the dependency still precedes that
   micro-batch's text forward. Mirror it for backward.
4. Load the reordered IR and re-simulate.
5. **Pass/fail: the bubble count must fall below 2019.** No wall clock, no PCIe
   dependence -- the same number would be 2019 on any fabric, so a reduction is a
   property of the schedule rather than of this box.

## Status, corrected

| report element | state |
|---|---|
| CP: single large image split on the patch dimension, gather-KV | **done, exact** (fp32 `max\|delta\| = 0` at 2 ranks, 4 grids incl. video) |
| CP: sub-CP groups, load-balanced large images | **done, exercised** (2 layouts same loss; 2/1 split runs the empty pass) |
| DEP: ViT and text as separate stages | **done, exact** (pp2/1F1B, pp4/1F1B, pp2/Interleaved1F1B) |
| DEP: vision passes balanced across PP stages | **not done** -- 64 vision slots on rank 0, 0 on the other seven |
| DEP: first micro-batches upfront, rest into bubbles | **not done**, and the route I assumed would deliver it does not |
| "most of the ViT computation is hidden" | **no evidence, and currently false** by the bubble count |

Two of my own claims in this document were wrong and are corrected above rather than
edited away: that a stage's actions would be placed into bubbles, and that the
decomposition was by depth. Both were assumptions that a measurement or a careful
re-read overturned.

---

# The reorder is INFEASIBLE, and this closes a loop I had opened myself

## The measurement

24 reorder candidates -- ``keep_first`` in {0,1,2,4} x ``lookahead`` in
{1,2,4,8,16,32}, forwards deferred and backwards hoisted, action multiset verified
preserved every time. **All 24 were rejected by the lowering pass**: 19 with
``AssertionError: Malformed compute schedule, can't schedule sends/recvs`` and 5 with
a rank-specific variant. Zero produced a schedule at all, let alone a smaller bubble
count.

So the compute IR is not freely reorderable. The lowering requires the per-rank
compute order to admit a consistent send/recv pairing, and moving a stage's forward
later on ONE rank breaks it: the consuming rank's forward for that micro-batch still
sits earlier in its own list, so the send cannot precede the receive.

Making it feasible would mean deferring the consumer too, which cascades through
every downstream stage -- that is not a reorder of this schedule, it is a different
schedule.

## Why that is not a tooling limitation

Because at ``n_vit = 1`` **the vision stage is the pipeline HEAD**. Every text stage's
first forward waits on it. The bubbles are downstream of the work that would fill
them, so deferring vision work delays the pipeline rather than hiding inside it.

This is exactly the argument in ``VIT_DEP_DESIGN_2026-08-06.md``, which I wrote and
then set aside:

> Stage 0 needs microbatch 0's vision features at the START of its first forward.
> Stages 1..P-1 are idle at that moment precisely because they are waiting for stage
> 0. The idle that looks free is upstream of the work that would fill it.

I superseded that on the grounds that it came from my own timing analysis rather than
from the report. The analysis was right. Three independent measurements now say so:
the bubble count is unchanged by DEP (2019 either way); every reorder is rejected as
infeasible; and the dependency direction is what makes both true.

**The lesson is about the supersede, not the analysis.** "This came from my own
reasoning rather than the source" is a reason to CHECK a claim, not to discard it.
Discarding it cost a design round.

## What is left

The report's mechanism cannot be the ViT sitting in the pipeline's dependency chain,
because anything in the chain is on the critical path by construction. It has to be
the ViT running CONCURRENTLY with the text pipeline -- micro-batch m's encode
overlapping micro-batch m-k's text compute on the same device, on a separate stream --
with the pipeline never waiting on it because it ran k micro-batches ahead.

That is outside the PP schedule entirely, which has two consequences worth stating
before anyone tries it:

* The AttnRes adapter's property that "PP owns all NCCL" would have to be given up
  deliberately, not incidentally. A vision collective issued from a side stream
  while PP is mid-schedule is exactly the shape that deadlocks.
* The lookahead k is bounded by memory: k micro-batches' worth of vision features
  must be held live. At ``max_images_per_batch`` and the feature widths involved that
  is a real budget, and it is the thing that decides whether "most of the ViT
  computation" can be hidden or only some of it.

## Status, final for this stretch

| report element | state |
|---|---|
| CP: patch-dimension split, gather-KV | done, exact |
| CP: sub-CP groups, load balance | done, exercised |
| DEP: ViT and text as separate stages | **done, exact** at pp2/1F1B, pp4/1F1B, pp2/Interleaved1F1B |
| DEP: passes balanced across PP stages | not done |
| DEP: first upfront, rest into bubbles | **not achievable this way** -- proven, not assumed |
| "most of the ViT computation is hidden" | false as built; needs a concurrent-stream design |

What DEP delivers today is the decoupling: the tower's placement is configuration
rather than a hard attachment to the embed stage, and it is numerically exact. That is
a prerequisite for the concurrent design and worth keeping. It is not the bubble
hiding, and the docs no longer suggest it is.
