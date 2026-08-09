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
| DEP: vision passes balanced across PP stages | **not done** -- 64 vision slots on rank 0, 0 on the other seven. Superseded: see the last section -- the clause is real, worth ~1/n in unhidden cost, and `KIMI_VIT_DEP_STAGES=2` hard-fails today |
| DEP: first micro-batches upfront, rest into bubbles | **not done**, and the route I assumed would deliver it does not. Superseded: the run-ahead delivers the mechanism (31/32 encodes issued early, verified); the hideable SHARE is what the split governs |
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
| DEP: passes balanced across PP stages | not done -- quantified and diagnosed in the final section |
| DEP: first upfront, rest into bubbles | **not achievable this way** -- proven, not assumed |
| "most of the ViT computation is hidden" | false as built; needs a concurrent-stream design |

What DEP delivers today is the decoupling: the tower's placement is configuration
rather than a hard attachment to the embed stage, and it is numerically exact. That is
a prerequisite for the concurrent design and worth keeping. It is not the bubble
hiding, and the docs no longer suggest it is.

---

# The run-ahead engages (2026-08-08 late): two bugs, and a falsifiable hit count

The previous section ended with the run-ahead as the only route to the report's
"remaining forward passes are scheduled into pipeline bubbles". It was written and
then reported as not engaging: `KIMI_VIT_PREFETCH=1` gave loss and grad_norm
identical to the off arm with the install log line absent from BOTH arms, which is
the signature of a path that never ran rather than evidence about the prefetch.

## Bug 1: the install call was in the pipelining_fn nothing uses

`_install_vision_prefetch` was called from `pipeline_llm_with_cache_adapter`. Every
kimi_k3 flavor registers `pipeline_kimi_k3_with_cache_adapter` instead
(`__init__.py` plus eight sites in `config_registry.py`), so the call was dead code.
No `isinstance` subtlety was involved -- the earlier guess that `model_parts` held a
type the check missed was wrong, and the diagnostic that would have printed those
types was itself downstream of a function that never ran.

## Bug 2: `logger` was not imported, and only the dead line used it

With the call added, rank 0 found the vision stage, patched it, and then died on
`NameError: name 'logger' is not defined` at the install log line -- the module used
`logger` in exactly one place, the line that had never executed. A latent NameError
sitting behind an uncalled branch is the same failure class as the dead call above:
neither is visible until something runs.

## The diagnostic had to be made rank-aware, not just present

The first version warned whenever a rank had no `KimiK3ViTStage`. Under DEP the
vision stage is global stage 0, so exactly one rank holds it and every other rank
legitimately has none: the warning fired on a CORRECT run. It now warns only on a
rank that owns stage 0 and still found no vision stage -- the genuine
misconfiguration -- and logs at info level elsewhere, so absence stays assertable
without crying wolf.

## Engagement, and a count predicted BEFORE it was measured

`installed` alone only proves the patch is in place. If `take(mb)` missed every time,
the encode would fall back to the synchronous path and the numbers would be identical
for the second time, for a different reason. So the prefetcher now counts hits and
misses and reports them per step.

At pp2, `local_batch_size=4` (n_microbatches = 4), depth 1, the count is derivable in
advance: mb0 must miss because nothing has been encoded yet, and mb1..mb3 must hit.
**Predicted 3 hits / 1 miss; measured 3 hits / 1 miss**, on every one of the five
reports.

| arm | installed | hits/misses per step | loss | grad_norm |
|---|---|---|---|---|
| `KIMI_VIT_PREFETCH=0` | 0 | -- | 12.07786 | 12.5625 |
| `KIMI_VIT_PREFETCH=1` | 1 | **3 hit / 1 miss** | 12.07786 | 12.5625 |

Five reports rather than two also checks out: `global_batch_size 8 / local 4 / dp 1`
is two `schedule.step` calls per training step, so 3 steps is 6 calls, and the first
has nothing to report. The mb0 miss is not a defect either -- it is the report's own
"the ViT forward passes of the first PP micro-batches are executed synchronously
upfront".

**Now the identical loss means something.** The same two numbers were previously
worthless; the difference is entirely that engagement is asserted independently, from
a count whose expected value came from the configuration rather than from the
measurement.

## The pp8xvp4 latency A/B: run, and NOT decidable on this box

Asked for explicitly: pp8 x vp4 multimodal, total latency, DEP toggled by one extra
parameter. Done, with the negative control the structural analysis made necessary.
Latency is read from the log's own millisecond timestamps across steps 10..20 (10
measured steps, `log_freq` 10's natural print points, steps 1-9 as warmup),
determinism OFF because deterministic algorithms serialise the concurrency being
measured.

| arm | ms/step |
|---|---|
| base (ViT synchronous on the embed stage) | 2063.3 |
| dep_nopf (DEP + side stream, run-ahead OFF) -- negative control | 2095.6 |
| dep_pf (DEP + side stream, run-ahead ON, 31 hit / 1 miss) | 2072.6 |

Repeated three times, arms INTERLEAVED so a drift over the half hour could not land
on one arm:

| repeat | dep_nopf | dep_pf | pf relative |
|---|---|---|---|
| initial | 2095.6 | 2072.6 | -1.1% |
| r1 | 2102.9 | 2088.3 | -0.7% |
| r2 | 2059.8 | 2121.2 | **+3.0%** |
| r3 | 2101.5 | 2063.5 | -1.8% |

Medians 2101.5 vs 2088.3, i.e. -0.6%; **spread WITHIN dep_nopf is 2.1% and within
dep_pf 2.8%, the ranges overlap completely, and the sign flips between repeats.** The
measurement does not resolve the effect.

### And that was predictable before running it

Two independent reasons, both hardware-facts rather than opinions:

* **mfu is 0.12% in all three arms.** So ~2.5ms of a 2063ms step is GPU compute and
  the remaining 99.88% is PP communication, Python and data loading. Hiding ALL vision
  compute could remove at most ~0.12% of the step, while the arm-to-arm spread is
  1.1-3.0% -- an order of magnitude larger.
* **The theoretical hideable share at this flavor is ZERO** (next section). So a
  CORRECT implementation must also show no improvement here. The absence of a win is
  not evidence against the implementation, and a win would have needed explaining.

Reportable: "no resolvable step-latency difference at pp8xvp4 on 8x5060Ti/PCIe, with
run-to-run spread 2-3x the effect ceiling". NOT reportable in either direction: any
claim about how much ViT computation is hidden.

## What IS decidable without a clock: `dep_hiding_theory.py`

The Interleaved1F1B action order depends only on `(pp_size, num_stages,
n_microbatches)`, so it can be built in one CPU process with no model and no GPU --
which is the whole point, since a number derived from an 8-rank run here would carry
this box's timings with it.

The analysis walks the vision rank's action list keeping a bubble budget: only bubbles
EARLIER than a micro-batch's consumption point can pay for its encode, because a
bubble after the consumer has already run is worthless however large the total. Cost
is in units of one text-stage forward, so the single model-dependent input is
`r = (one ViT forward) / (one text-stage forward)`.

| r | hidden | limited by |
|---|---|---|
| 0.01 - 0.3 | **18/32 = 56%** | TIMING (uses only 2.2% of bubble capacity) |
| 1.0 | 8/32 = 25% | mixed |
| 3.0 | 2/32 = 6% | capacity |
| >= 10 (incl. the debug flavor's 25.2) | **0/32 = 0%** | capacity |

* **debug `report_arch_pp8vp4`, r = 25.2: 0% hideable.** One ViT forward costs 25.2
  units and the bubbles accumulated before any vision forward never reach that. This
  is why the latency A/B could not have shown anything.
* **real `2p8t_vl`, r = 0.057: 56% hideable**, consuming 2.2% of bubble capacity. The
  binding constraint is TIMING, not capacity: the first payable encode is at slot 46,
  so the first 14 micro-batches fall in the warmup stretch where too few bubbles have
  accumulated -- which is the report's own "the ViT forward passes of the first PP
  micro-batches are executed synchronously upfront", arrived at from the schedule
  rather than read off the page.

The r range matters because r is a DATA parameter (visual tokens per micro-batch), not
a model constant, so a single value would have been a single-point claim.

### A modelling mistake worth recording

The first version of this analysis demanded that EVERY encode be covered by earlier
bubbles, and duly reported TIMING DEFICIT at vision forward #2 -- **it failed a
schedule that matches the report exactly**, because the report itself runs the first
micro-batches synchronously. The question was wrong: not "can all of it be hidden" but
"how much can be". A pass/fail oracle on a quantity the source describes as partial
will always answer fail.

### This also corrects a claim I made in the PP adapter analysis

`PP_ADAPTER_UNDER_SIDESTREAM_VIT_2026-08-08.md` argued that a same-box relative A/B
"measures whether this implementation hides work on this hardware, which is a
different and answerable question". Legitimate in principle, but not answerable when
the effect ceiling (0.12%) sits an order of magnitude below run-to-run variation
(2-3%). The original instruction -- do not measure wall-clock, use theoretical
calculation, because this PCIe box's communication latency does not extrapolate -- was
more thoroughly right than the caveat I attached to it.

---

# Clause (2) revisited: "balanced across PP stages" is real, quantified, and unbuilt

## I had talked myself out of the report's own words

The verbatim sentence is:

> splits ViT and text training into separate stages **and balances vision forward and
> backward passes across PP stages**

This document's earlier section read that as "the ViT occupies more than one stage, by
definition, not as a later optimisation". A later section replaced it with "distributing
the per-micro-batch vision passes over the stages' idle slots", and the status table
went to **not done**.

That replacement was a mistake, and the tell was available at the time: the substitute
reading is **exactly the thing already proven impossible** -- all 24 action reorders
rejected by the lowering. Choosing an interpretation under which the report describes
something unimplementable, when a straightforward implementable reading exists, is
choosing the wrong interpretation.

**The two clauses are not rivals.** "Balances ... across PP stages" (the tower spans
several stages) and "further decompose the ViT computation" (per micro-batch: first
ones upfront, rest into bubbles) are separate requirements in separate sentences. I
collapsed them into an either/or and retired the `n_vit > 1` work on that basis. This
is the SECOND time tonight the same error shape appeared -- see also superseding my own
correct 2026-08-06 dependency analysis. The pattern: **when two readings look like
they conflict, check whether the source is asking for both.**

## What the split is worth, computed rather than argued

`dep_hiding_theory.py --split` puts vision stage i on rank `i % pp`, so n stages spread
the work over that many ranks and each carries `r / n`. At pp8 x vp4, 32 micro-batches:

| n | r/n | ranks | worst-rank hidden | all-rank hidden | worst UNHIDDEN cost |
|---|---|---|---|---|---|
| 1 | 0.057 | 1 | 56% | 56% | 0.798 |
| 2 | 0.029 | 2 | 56% | 78% | 0.399 |
| 4 | 0.014 | 4 | 56% | 89% | 0.200 |
| 8 | 0.007 | 8 | 56% | **95%** | **0.100** |
| 16 | 0.004 | 8 | 66% | 96% | 0.078 |

At the debug ratio r = 25.2 the same split moves the hideable share from 0% to 8% and
the unhidden cost from 806 to 94.5 units.

**The share is the wrong metric and it took a wrong turn to notice.** Worst-rank hidden
share sits at 56% for n = 1..8 and looks like the split does nothing. It does not move
because rank 0 is the pipeline head and has no bubbles during warmup regardless of the
split. What changes is how much vision work that rank carries at all, so the metric
that tracks wall time is the **unhidden cost**, and that falls as ~1/n. A ratio that
holds steady while its numerator and denominator both shrink hides the entire effect.

The all-rank 95% at n = 8 is also the first number in this work that lands where the
report's "most of the ViT computation is hidden" would put it -- reached from the
schedule, at the real model's cost ratio, with no clock involved.

## Status of clause (2): hard-fails today, and that is the good failure mode

`KIMI_VIT_DEP_STAGES` exists and defaults to 1. Setting it to 2 at pp4:

```
STAGES=1  exit=0  loss 12.07055
STAGES=2  exit=1  ValueError: Optimizer param_groups pattern '.*' matched no parameters
```

Cause, confirmed by reading and then by the run: `_inject_kimi_k3_fqns` emits n vision
FQNs, but `_parallelize_with_tower` only converts the chunk that holds `embed_tokens`
and no layers into a `KimiK3ViTStage`. Vision chunks 1..n-1 have neither, so they fall
through to the plain-text branch as zero-parameter shells and the optimizer finds
nothing. **It fails loudly rather than training a hollow stage**, which is the failure
mode to want -- unlike tonight's two silent ones.

## What building it actually requires

Not a small change, and specifically not "loosen the detection":

1. **Split MoonViT's encoder layers across the vision stages**, the way the text stack
   is split -- first stage keeps `patch_embed` (plus `embed_tokens`), last keeps the
   projector and the splice, middles keep layer ranges.
2. **Thread the CP patch plan through stage boundaries.** `set_cp_patch_plan` currently
   hands one plan to one tower; with the tower spanning stages each piece needs its
   own, and dynamic CP's gather-KV is issued inside the attention it splits.
3. **Redefine the run-ahead.** It currently prefetches by calling `encode_images`,
   which assumes one stage does the whole encode. Across stages an "encode" becomes a
   pipeline of its own, and what runs ahead is the FIRST vision stage's share while
   later shares arrive over the pipe -- a different mechanism from the present one, not
   a parameter change.

Deliberately not started tonight: the run-ahead and the stage split are both exact
right now, and (3) rewrites the mechanism (1) and (2) would have to be validated
against. Each half needs its own numerical gate, from a shared seed checkpoint, or a
regression in one will be read as a defect in the other.

## Clause (2), step 1 of the build: the block split is numerically pinned

Done and committed ahead of any pipeline wiring, deliberately: a mismatch discovered
later at pp4 would be indistinguishable from a PP plumbing bug.

`MoonViTEncoder` gained two methods, and `forward` now delegates to the second so the
single-stage path is the same code:

* `block_inputs(x, grid_thws, cp_plan) -> (freqs_cis, cu_seqlens)` -- the per-forward
  values every block needs, split out so each stage can RECOMPUTE them from its own
  `grid_thws`. Recomputing rather than sending them is forced: PP's metadata inference
  pushes dummy values through pipe tensors, and these are RoPE indices and segment
  bounds, where a dummy asserts out of bounds. Same reason `input_ids` never leave the
  vision stage.
* `run_blocks(..., block_slice=..., apply_final_norm=...)` -- one contiguous share,
  with the final norm belonging to the last share only.

`tests/test_moonvit_stage_split.py`, fp32 at `rtol=1e-5` (loose tolerances have already
hidden a real 1e-3 defect in this model once):

| property | result |
|---|---|
| 2 shares chained == whole encoder | pass |
| 4 shares (one block each) == whole | pass |
| norm on EVERY share must NOT match | pass -- guards the test itself |
| `block_inputs` identical when recomputed on a later share | pass, exact (rtol=0) |
| gradient reaches blocks in BOTH shares | pass |

The third row is the one that keeps the suite honest: without it, a test that passed
regardless of where the norm went would not be testing the split at all.

## Clause (2), step 2: a constraint found while designing, not while coding

Splitting the tower collides with WHERE THE SPLICE HAPPENS, and the collision is not
obvious until the data flow is written out.

* The splice needs `input_ids` -- to locate the sentinels and to embed the non-visual
  tokens.
* `input_ids` can only be on the FIRST pipeline stage: torchtitan passes positional
  args to stage 0 only, and ids cannot travel the pipe (dummy values indexed into an
  embedding assert out of bounds -- measured, in an earlier round).
* But with the tower split, the visual features are only ready on the LAST vision
  stage.

So "tower spans stages" and "splice needs ids" pull in opposite directions. The
resolution keeps every pipe payload a float activation:

1. **vision stage 0**: `patch_embed` + `blocks[0:k1]`, AND `embed_tokens(input_ids)`.
   It emits `(x_LD, text_embeds, sentinel_mask)`.
2. **middle vision stages**: their block share on `x_LD`, passing the other two
   through untouched.
3. **last vision stage**: remaining blocks + `final_layernorm` + merger + projector,
   then splice the features into `text_embeds` at `sentinel_mask`.

`text_embeds` is exactly the kind of payload the pipe already carries today (the
current single vision stage sends spliced embeddings), so dummy values in it are
harmless -- nothing indexes with them. `sentinel_mask` travels as a float mask for the
same reason.

**And an implementation boundary that has to be declared rather than discovered:** this
works for the `_splice_per_token` convention, where `MMCollator` reserves one sentinel
per post-merge visual token so the sequence length is already correct. It does NOT
work as-is for `_splice` (one sentinel per image, expanded in place), because that
convention CHANGES the sequence length per sample and right-pads to a common length --
a shape that depends on the batch's image token counts, which PP's shape inference
cannot be handed. The split must therefore reject the LLaVA-style convention
explicitly instead of producing a shape that works on one batch and not the next.

Step 2 is designed, not built. What it needs beyond the above: `_parallelize_with_tower`
has to identify WHICH vision chunk it is looking at (today it recognises only "holds
embed_tokens and no layers"), and the run-ahead has to be gated off for n > 1 until its
cross-stage form exists -- prefetching by calling `encode_images` assumes one stage does
the whole encode, so leaving it on would silently do the whole tower on stage 0 and
defeat the split it is supposed to support.

## Clause (2) is BUILT and numerically exact (2026-08-09)

From a shared seed checkpoint -- mandatory, since a different vision-stage count changes
the stage split and a cold start would compare initialisation rather than the split:

| configuration | loss | grad_norm | stages wired | roles |
|---|---|---|---|---|
| pp4 1F1B, `n_vit=1` | 12.07418 | 12.5000 | 1 | `both` |
| pp4 1F1B, `n_vit=2` | **12.07418** | **12.5000** | 2 | `head` / `tail` |
| pp4 Interleaved1F1B, `n_vit=2` | **12.07418** | **12.5000** | 2 | `head` / `tail` |
| pp4 Interleaved1F1B, `n_vit=4` | **12.07418** | **12.5000** | **4** | `head` / `body` x2 / `tail` |

Bit-identical across all four, and the vision shares land on DIFFERENT ranks -- which is
the clause. `n_vit=4` is the first configuration to exercise the `body` role under a real
pipeline. Four is this flavor's maximum because its tower has four blocks; `block_bounds`
raises above that rather than emitting an empty share.

Engagement is asserted from a log line (`DEP vision stage wiring: N stage(s), roles
[...]`), never from the loss: the split is numerically neutral by design, so loss alone
cannot distinguish "split" from "silently unsplit".

### Three obstacles, and what each turned out to be

**1. Chunk identity.** `_parallelize_with_tower` recognised the vision chunk as "holds
embed_tokens and no layers", which identifies share 0 only -- later shares hold neither
and look exactly like a text chunk. Call order is unusable: a rank holding several
virtual stages receives them in an order the function is not told. Fix: turn
`__kimi_dep_vision__{i}` from a deliberately-unmatched FQN into a real empty submodule.
`_split_module` keeps the child named in the chunk's FQN list, so the chunk arrives
carrying its own index. No parameters, so no stage gets heavier.

**2. A silent-failure path designed out.** The micro-batch index patch was installed only
when the run-ahead was on, and a split tower's later shares NEED it to find `grid_thw`.
Without it they take the metadata-inference branch: activations pass through unprocessed,
no tower, no splice, no error. It is now `_install_vision_stage_wiring`, installed
unconditionally under DEP and logged. Related: on a batch with no images the later shares
must still run a placeholder-sized tower, or they skip an FSDP2 all-gather their peers
issue and the peers wait for the watchdog. "No micro-batch in flight" and "micro-batch
has no images" are now separate paths.

**3. A premise of mine was wrong and the crash said so.** I assumed only the first stage
receives the batch's kwargs, and built `VisionStepInputs` plus a step hook so later shares
could recover `grid_thw`. The failure
`KimiK3ViTStage.forward() got multiple values for argument 'pixel_values'` says otherwise:
**PP forwards the batch kwargs to EVERY stage.** A later share therefore reads `grid_thw`
directly, with nothing extra on the wire and no cache; the cache survives only as a
fallback. `forward` now takes `*args` because a later share receives its upstream's three
tensors positionally AND the batch kwargs by name -- under a named signature the patch
stream binds to `input_ids` and `pixel_values` then arrives twice.

### Declared limits, not discovered ones

* **No CP with a split tower.** The shard decision and the dynamic-CP patch plan are made
  inside `encode_images` and every share would have to recompute them identically.
  `_dep_reject_cp` raises. `n_vit=1` with CP is unaffected.
* **Per-token collator convention only.** The one-sentinel-per-image convention changes
  sequence length per sample and right-pads to a batch-dependent length, which PP cannot
  size a buffer for. The tail raises rather than producing a shape that works on one
  batch and not the next.
* **`dep_max_images` is a budget over FRAMES**, since `t` has no configured maximum. A
  video exceeding capacity raises at the sender; truncating an activation would be a
  silently wrong model the receiver cannot detect.

### What the split is worth, restated correctly

Earlier in this document I attributed the 1/n gain to bubbles. That was wrong and the
correction stands: a vision share cannot start before its upstream share delivers, so
only the head can move earlier in time. **The split's gain is load balancing** -- vision
work on the critical path drops to r/n because it is spread over n ranks -- and that needs
no bubbles at all. `dep_hiding_theory.py --split` reports the right 1/n number for this
simpler reason.

Which is why this clause's verification target is **peak memory and per-rank vision load,
not latency**: at the real cost ratio (r ~ 0.057, MoonViT's 401M being an order of
magnitude under a text stage's ~3.4B activated) the time on the table is under a percent
of a step, while moving the tower's parameters and activations off one stage is
unconditional.

### The memory verification FAILED its prediction, and that corrects my own advice

I recommended peak memory as this clause's verification target and called the gain
"unconditional in the cost ratio". Measured at pp4 x Interleaved1F1B, 3 steps, cold start:

| n_vit | per-rank peak GiB (desc) | max | vs n=1 |
|---|---|---|---|
| 1 | 0.99 / 0.58 / 0.33 / 0.27 | 0.99 | -- |
| 2 | 1.05 / 0.66 / 0.30 / 0.27 | **1.05** | **+6%** |
| 4 | 0.93 / 0.64 / 0.45 / 0.41 | 0.93 | -6% |

**Not monotone, and n=2 is WORSE than not splitting.** The distribution does even out --
the lightest rank goes 0.27 -> 0.41 GiB, so the parameter-level balancing is real -- but
the maximum, which is what a rank must fit, barely moves and moves the wrong way first.

The likely cause is the fixed-capacity pipe payload the split requires:
``stage_patch_capacity(32, 32, 8)`` is 8192 rows, times vision hidden 256, times 2 bytes
= 4.19 MB per tensor, times the micro-batches in flight and both directions. That lands
in the same tens-of-MB range as the +64 MB seen at n=2. Meanwhile this flavor's tower is
four blocks at hidden 256 -- a few MB -- so the split saves less than the padding costs.

**I am not fitting a decomposition to two points.** Solving
``net(n) = -T(1 - 1/n) + P(n - 1)`` against the two measurements gives a NEGATIVE tower
activation, which means the model is wrong: the padding cost is not simply linear in the
number of boundaries, and allocator peak timing and fragmentation are in the number too.
Two points do not support a quantitative split of the effect.

**So the correction to my own advice:** peak memory is not unconditional. It is
``tower_size / n`` saved against ``padding_capacity`` spent, and on this flavor that ratio
is unfavourable. The claim that survives is narrower: **the split balances the vision
parameters across ranks, and whether that reduces the peak depends on the tower being
large relative to the configured patch capacity.**

Which is the same shape as the cost-weight lesson elsewhere in this work: a debug flavor's
tower is ~5 MB against the real MoonViT's 401M, so this measurement does not extrapolate.
**And a correction to the paragraph above, made by checking instead of assuming.** I first
wrote that 8/32/32 over-provisions the capacity by ~8x. It does not: the collator's
``max_patches`` is 1024 per image, i.e. exactly a 32x32 grid, and ``local_batch_size=8``
with one image per sample is exactly 8 images. **The configured capacity matches the
collator's worst case precisely.** It cannot be lowered without lowering what the data is
allowed to contain.

That makes the conclusion stronger and less comfortable: the padding is an INHERENT cost
of splitting, not a misconfiguration. PP sizes its buffers once, so a mid-tower boundary
must reserve the worst case every step even when a batch uses less, and that reservation
is independent of how large the tower is. So splitting a SMALL tower is a net memory loss
by construction, and the break-even is set by tower size against worst-case patch budget.

What would actually decide it for the real model: MoonViT is 401M against this flavor's
~5 MB of vision blocks, roughly five orders of magnitude, while the patch capacity grows
only with the visual token budget. The sign of the effect is therefore expected to flip at
real scale, but that is a prediction from the structure and not a measurement -- and the
same cost-weight caution applies as everywhere else in this work. Until it is measured on
a flavor whose tower is not negligible, clause 2's demonstrated result is EXACTNESS and
PARAMETER BALANCING, not a memory win.

---

# Dynamic CP is DEP's PRECONDITION, not a sibling optimisation (2026-08-09)

Found by re-reading 5.2.3 for the pipeline degrees. The Dynamic CP paragraph does not end
on load balance -- it ends on this:

> This reduces both the encoder latency of large visual samples and the cross-device load
> imbalance, **allowing the remaining encoder computation to be hidden in pipeline
> bubbles.**

So the report's causal chain is: large images/videos make the vision cost large -> too
large for the bubbles -> Dynamic CP partitions the patch dimension across CP ranks ->
the REMAINDER fits. The two subsections are one mechanism, and clause (3) depends on the
Dynamic CP work rather than sitting beside it.

## Why r is large for big images: the quadratic term I had omitted

My earlier estimate of `r` (one ViT forward in units of one text-stage forward) used only
the linear term, `401M x n_patches`. Vision attention is quadratic in patch count, so

    r ~ [ 401M * n_p  +  n_p^2 * heads * head_dim * 27 ] / (text_stage_activated * seq_len)

and at the patch counts long-context multimodal training reaches, the quadratic term
dominates. That is what "large images and long videos substantially increase the
computation time of the vision encoder" refers to, and it is why the linear estimate made
DEP look pointless at scale.

Dynamic CP reduces BOTH terms by `cp_size`: gather-KV has each rank compute `n_p/cp`
queries against all `n_p` keys, so the quadratic work per rank is `n_p^2/cp`, not
`(n_p/cp)^2`. Hence `r_eff = r / cp_size` for the hiding analysis.

## The threshold, and what it says about this box

From `dep_hiding_theory.py --sweep`, reaching the report's "most of the ViT computation is
hidden" (>50%) needs **r_eff <= 0.3**. So the required CP degree is `cp >= r / 0.3`:

| scenario | r | cp needed for "most hidden" |
|---|---|---|
| `report_arch_pp8vp4` (debug) | 25.2 | **>= 84** -- impossible on 8 GPUs |
| a mid-size large image | 2.4 | 8 |
| real `2p8t_vl`, one 1024-patch image | 0.057 | 1 (already 56%) |

Measured directly at pp8 x vp4, r_eff = 25.2/cp: cp=1 and cp=2 give 0% hideable, cp=4
gives 3.1%, cp=8 gives 6.2%, cp=16 gives 15.6%.

**This is the third independent confirmation of the same thing.** The debug flavor cannot
show bubble hiding, and it is now established by three routes that do not share an
assumption: structurally (the vision stage is the pipeline head, so the bubbles are
downstream of the work), theoretically (r=25.2 needs cp>=84), and empirically (2019 = 2019
bubbles, and a latency A/B whose sign flips between repeats).

## What this implies for measuring the effect at all

The knob that matters is `r`, and it is NOT the layer-count ratio. Those are already
aligned: 13 text / 4 vision here against the report's 93 / 27, i.e. 3.25 vs 3.44. What
differs by ~440x is the PER-LAYER cost ratio -- the report's text layers are ~1.12B
activated each (104.2B / 93) against MoonViT's ~15M (401M / 27), about 75x apart, while
this flavor has hidden 256 on both sides so its layers cost about the same.

So a flavor built to observe bubble hiding on 8 GPUs needs `r_eff <= 0.3`, reachable by
shrinking the vision tower relative to the text stack (roughly vision hidden at 1/8 of
text hidden) and/or leaning on CP. That does not require the real 2.8T model, only a
config whose cost ratio is honest.

## And the report does NOT give the pipeline degrees

Checked, because our pp8 x vp4 has been described as the report's configuration in
passing. Section 5.2 says only:

> Kimi K3 pre-training combines Pipeline Parallelism (PP) with virtual stages (VP) [48,
> 81], Expert Parallelism (EP) [66], ZeRO-1 Data Parallelism [100], Pipeline ZeRO-2
> gradient sharding [145], and Context Parallelism (CP, 5.1.2) [50].

There is no PP degree, no VP count and no micro-batch count anywhere in the paper (896 is
the routed-expert pool, not an EP degree). **pp8 x vp4 is our choice and must be described
as ours**, not as the report's topology.
