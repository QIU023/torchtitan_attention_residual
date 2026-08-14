# The migration plan, corrected -- read this before the older step docs

Two things were wrong in how the migration was being run today, and both were caught by
the human rather than by me.

## 1. Every matrix result must name the tree it ran on

I reported "54/54" for the carrier migration. True, and measured on **our** tree
(`torchtitan/models/kimi_k3/`), which already passed 54/54 before any of this. It says
the upstream carrier shape did not break our tree. It says **nothing** about migration
progress.

The upstream tree (`torchtitan/models/kimi_k3_up/`) is at **0/54** and cannot reach 54
today:

| the upstream tree needs | state | cells it gates |
|---|---|---|
| FSDP | shipped | baseline |
| CP (Ulysses + KCP) | implemented, a few cells checked | 5 |
| PP (`pipelining_fn` + carrier) | wired, **never run successfully** | 8 |
| EP | **absent** | 6 |
| TP | **absent** | 8 |
| **LoRA** | **absent** | the whole `mm_lora` arm, 18 |

LoRA was previously filed as "part of step 3, can wait". Wrong: it is a **precondition**
for 54/54 on that tree, since one of the three arms is a LoRA flavor. Five workstreams,
not four.

`mm_full` needs no new flavor -- their `debugmodel` is our `report_arch` in every free
parameter.

## 2. Order: finish declarative in OUR tree, then swap modules

Measured rather than argued.

Remaining imperative surface in our tree:

* `parallelize_module`: **4 real call sites** (line 704 vision tower, 861, 882, 1120 --
  the other two greps are an import and a comment)
* `use_local_output`: 38
* modules already carrying a `sharding_config`: **29**

Two candidate orders:

**A -- port each parallelism onto their tree** (what today was doing). Every one of the
first four steps runs on a tree that **cannot express the three-arm matrix**, so each is
gated on "it trains" rather than on 54/54. Today's error rate was concentrated exactly
where a criterion was missing or invalid.

**B -- finish declarative in our tree, then swap module implementations.** Every step runs
where 54/54 is expressible. And once sharding is declared on the config tree, swapping a
module **carries its parallelism with it** instead of needing the parallelism re-ported.

**B is faster, and not because it is less work.** A's first four steps have no gate, and
an ungated step produces conclusions that get redone.

The carrier migration is the worked example of B: adopt their `[T, N, D]` threaded
carrier inside our model, keep our adapter and our parallelism, gate on 54/54 plus a
bitwise cache-on/off comparison. It landed in one pass.

What blocked B before today is now gone: the KIND probe located the residual-stream kind
mixing, and the carrier change removed it -- a Python list cannot be declared, a threaded
tensor can.

### Order

1. **Declarative in our tree**: the 4 `parallelize_module` sites and the 38
   `use_local_output` sites, each step gated on 54/54 (our tree).
2. **Swap modules to the upstream classes**, one at a time, each gated on 54/54:
   `KimiMoE` -> `KimiLatentMoE` + `KimiGroupedExperts` (also the only way to get
   `router_input_BLD` out of `common/moe.py`), then the attention classes, then the
   vision encoder.
3. When the swap is complete the migration is done: our tree IS their structure, with
   declarative parallelism that travels with the modules.

### What the work on their tree is still worth

Not wasted, just out of order. The Ulysses/KCP attachment patterns, the PP carrier's
bitwise verification, and the thin `pipelining_fn` all apply at step 2 -- when the modules
being wired are the ones that will actually ship.

---

## Step 1 started: which of the four imperative sites can move, and why

Branch `migrate_tp_attnres_tail`. Measured, not reasoned.

### The rule that decides it

A module can go declarative in isolation **iff its parameter is read at a use site that
already unwraps a DTensor**. If the module is CALLED (`mod(x)`), a declared weight meets
the plain residual stream inside the op and dies with `aten.mul.Tensor got mixed
torch.Tensor and DTensor` -- the declarative vocabulary has `state_shardings`, `in_src`,
`in_dst`, `out_src`, `out_dst` and `local_map`, and **none of them is `to_local` on the
output**. That is why `use_local_output=True` has no declarative equivalent.

Demonstrated on the AttnRes tail, whose two modules both already carried
`sharding_config=_tp_replicate()` (the imperative `NoParallel` plan was making the
declarative driver skip the subtree, so the declaration was inert):

| declared | result on `tp2` |
|---|---|
| `final_attn_res_proj` + `final_attn_res_norm` | **FAILS** -- mixed Tensor and DTensor |
| `final_attn_res_proj` only, norm left imperative | **trains**, step 1 identical to baseline, later steps 1e-5 |

`proj.weight` is read directly inside `block_attn_res`, which already calls `to_local()`
on it. `norm` is invoked as a module. Same rule, opposite outcomes.

### What this implies for the remaining sites

| site | nature | movable alone? |
|---|---|---|
| L704 vision tower | 92-line `_apply_tp_moonvit_mlp`, its own CP mechanism | out of scope for this migration |
| L861 model level | `embed_tokens` / `lm_head` | embeddings switch to VOCAB-PARALLEL execution once `parallelize()` sets `tp_group` -- a different mechanism, established earlier at the cost of 29 test failures |
| L882 AttnRes tail | two modules | **proj yes, norm no** (measured above) |
| L1120 LoRA TP | per-layer plan with `.base` keys | untested |

So the 38 `use_local_output` sites are not 38 independent edits. Each one is a module
whose output feeds the residual stream, and **the stream has to flip together** -- which
is the same conclusion the KIND probe reached from the other direction. The tensor carrier
made that flip expressible; it did not perform it.

### Honest state of step 1

One of four sites is half-migrated on a WIP branch, verified on `tp2` only, not on 54/54.
The 1e-5 drift on later steps wants explaining before this is called done: a declared
Replicate and `NoParallel(use_local_output=True)` should be the same placement, so the
difference is a routing detail and not obviously benign.

### The 1e-5 on the declared proj: two hypotheses refuted, one contradiction open

Declaring `final_attn_res_proj` (norm left imperative) trains and matches the baseline at
step 1, then drifts ~1e-5. A declared Replicate and `NoParallel(use_local_output=True)`
should be the same thing, so the drift needs a cause. Two were proposed and both are
dead:

* **Different mesh or placement.** Measured with a probe on `block_attn_res`'s operands:
  both `proj.weight` and `norm.weight` are `mesh=('tp',) placements=(Replicate(),)`.
  Identical.
* **The imperative call suppressed the declarative driver.** The removed line was
  `parallelize_module(model, tp_mesh, {...})` on the model ROOT, which could have marked
  the root parallelized and made the driver skip everything. It did not: both trees log
  `entered parallelize() on 4 outermost Modules: {'ScaledDotProductAttention': 4}`.

That leaves a contradiction worth starting from: **the driver does not enter the tail
modules at all** (only 4 SDPA), yet the probe shows `proj.weight` as
`DTensor(tp, Replicate)`. Something else distributes it. Until that is identified, "the
proj is declared" is not established -- it may be getting its placement from a path
nobody has named, which would also explain a 1e-5 that no placement difference accounts
for.

Next action: find what distributes `final_attn_res_proj.weight`. `_drive_declarative_sharding`
returns the classes it entered, so instrument that return rather than the log line, and
check whether `AttnResProjection` is reached and skipped or never visited.

### The 1e-5 explained: 27 declarations that never existed

`AttnResProjection` was constructed by calling the class:

    self.attn_res_proj = AttnResProjection(proj_cfg)

`_sharding_config` is assigned inside `Config.build()`, not in any `__init__`, so calling
the class **drops the declaration at construction**. All 27 AttnRes pseudo-queries in the
model were built that way, and the comment above them read "declared here so the module
carries its own placement like every other linear after the migration" -- describing an
intent the code never carried out.

Their placement came from the **final sweep** at the end of `apply_tp`, which promotes
every remaining plain parameter to `DTensor(tp, Replicate)`. So from outside the placement
was correct and the matrix was green; what differed between the two trees was not the
placement but WHO produced it, and at what point in the sequence. That is the 1e-5.

Fixed to `proj_cfg.build()` in three places; 27/27 now carry the declaration. An AST scan
for the same pattern -- both `Class(Class.Config(...))` inline and `cfg = ...Config(...)`
then `Class(cfg)` -- returns **no other occurrences**.

**Why this one matters more than its symptom.** Nothing raises, no placement is wrong, and
every matrix cell passes. The only observable effect is that the declarative migration
cannot advance: delete an imperative site and the corresponding declaration is empty, so
the parameter silently falls through to the sweep. It would have made every remaining step
of step 1 look inexplicable.

**Method note.** Three hypotheses, and the cheapest one was tried last. Two 8-GPU runs
went to "different mesh" and "the driver was suppressed"; the one that landed was a
CPU-only check of `_sharding_config` on a meta-device build. When the question is "is this
declaration active", build the model and look -- do not infer it from a training run.

### The declarative layer had never run, and it was right all along

`apply_tp`'s final sweep promotes every remaining plain parameter to
`DTensor(tp, Replicate)`, and it runs INSIDE `apply_tp` -- **before**
`_drive_declarative_sharding`. So the sweep did the declarations' work first and the
driver found every subtree already distributed.

Two things that made no sense now do:

* the driver only ever reported `{'ScaledDotProductAttention': 4}` -- the only declared
  modules with no parameters of their own for the sweep to claim;
* fixing 27 dropped `AttnResProjection` declarations changed no number at all.

With the sweep skipping declared modules, the driver enters **300**: 118 RMSNorm, 77
Linear, 43 AttnResProjection, 21 KimiMLP, 20 MoE, 15 KimiDeltaAttention, 6 SDPA.

**Result: 54/54 on our tree, all three arms.** Three hundred declarations that had never
executed were correct on their first run, across every parallelism combination the matrix
covers.

### What this does to the estimate of step 1

"4 `parallelize_module` sites plus 38 `use_local_output` sites" counted imperative LINES
on a tree where every declaration was inert. The declarations already covered 300 modules.
So step 1 is not mostly writing declarations -- it is removing the imperative pieces that
shadow them, and each removal is now a real handover rather than a silent fall-through to
the sweep.

### Method note, the sharpest of the day

The two checks that found this were both seconds of CPU: build the model on a meta device
and read `_sharding_config`, and look at which function calls which. Two 8-GPU runs went to
refuted hypotheses first. **When the question is "is this mechanism active", inspect the
built object -- do not infer it from a training run.**

### DEP with two vision stages: first run, and it works

`vit_dep_stages` had only ever been 1. Report 5.2.3 wants vision forward and backward
balanced ACROSS pipeline stages, the code supports it (`KimiK3ViTStage.set_dep_role` --
share 0 takes `patch_embed` and `embed_tokens`, the last share takes the projector and
the splice), and its own docstring says the value "needs measurement to set". No matrix
cell had ever set it above 1.

    pp4  dep_stages=2: 12.06363 ... 9.91233
    pp8  dep_stages=2: 12.04840 ... 9.61217

Both train 10/10. Two constraints had to be satisfied first, and both cost a run:

* **two vision stages need `pp_degree >= 4`.** They come OUT of the text budget rather
  than being added on top, so `pp2` has nothing left -- "DEP wants 2 vision stage(s) but
  only 2 stages exist". That constraint is in `dep_vision_stages()`'s docstring, which I
  had quoted one message earlier and then picked `pp2` cells anyway.
* **microbatches must be >= stages**, so `--training.local-batch-size 2` fails at pp4.

Neither was a DEP defect.

### Per-layer AttnRes: the rule applied, after being ignored once

Handing all four per-layer AttnRes modules to their declarations gives **33/54**, failing
exactly the `tp>1` cells in all three arms. The KIND probe: `input plain, weight
DTensor(R)` inside `rms_norm`.

That is the rule measured earlier the same day and then not applied -- a regex deleted all
four plan entries at once because it was convenient, when the rule says the projections
and the norms are not the same case. Projections declared, norms left imperative:
**54/54**.

So the residual-stream flip now has a quantified prize: it releases all 26 per-layer
norms, the tail norm, and the 38 `use_local_output` sites in one move.

### Is an all-DTensor residual stream what other models do? Yes -- it is their default

Counted, not assumed:

| model | `use_local_output` occurrences |
|---|---|
| llama3, deepseek_v3, qwen3, llama4, gpt_oss | **0 each** |
| kimi_k3 (ours) | **39** |

llama3's block is the whole argument:

    h = x + self.attention(self.attention_norm(x), attention_masks, positions)
    out = h + self.feed_forward(self.ffn_norm(h))
    return out

`x` arrives a DTensor, attention returns a DTensor, the adds happen on DTensors, `out`
leaves a DTensor. The stream is never unwrapped, and `llama3/sharding.py` declares its
layout directly -- SP on gives `Shard(1)` activations around attention and FFN with `wo`
and `w2` reduce-scattering into it; SP off keeps everything Replicate with an all-reduce
instead.

So the flip is not a scheme we have to invent. **A plain residual stream is our deviation**,
and it is the reason the declarative vocabulary has no output-side `to_local`: no upstream
model has ever needed one.

Two consequences:

* The remaining work is not "convert 79 modules". It is removing one deviation, and the
  26 per-layer norms, the tail norm, the 15 KDA layers and all 39 `use_local_output` sites
  come back together.
* It also de-risks step 2. The upstream modules assume a DTensor stream, so once ours is
  one they drop in without an adapter layer.

### The five tp x cp cells: measured, still open

`migrate_stream_dtensor` at `b063b9e09` is **49/54**. text 18/18; the five failures are
`fsdp2_tp2_cp2` and `ep2_fsdp2_tp2_cp2` on both multimodal arms plus `tp2_pp2_cp2` on
LoRA. They passed before the flip (10/10 in `mx_sweep`), so the flip introduced them.

**What is measured**, with a probe on the vision path rather than by reading:

    dynamic_cp returns: ['plain']     <- forward output is plain, as intended
    grad into feat[0]: DT             <- the gradient arrives as a DTensor

Forward plain, backward DTensor. The tower's hand-written `funcol.all_gather_tensor` has
a `reduce_scatter` transpose, and `_c10d_functional.reduce_scatter_tensor` has no DTensor
sharding strategy, so the backward dies there.

`to_local(grad_placements=...)` cannot express the fix: the tensor never came from a
DTensor, so there is nothing to unwrap on the way out. The constraint is only on the way
back.

**Six attempts, none complete.** In order: flip the splice's lift direction; stop lifting
the tower's input (-> `aten.convolution` mixed, because the tower's TP weights are
DTensors); pin `grad_placements` at the gather; restore `use_local_output=True` inside
`_apply_tp_moonvit_mlp` (-> `aten.add` mixed, tower output now plain while the stream
wants DTensor); an `autograd.Function` sealing the gradient at the tower's exit (->
`from_local` receives a DTensor, one layer further); scoping that seal to the entries
this path produced (same error).

The last two are the closest and the error moved, so the boundary is nearly right and the
remaining mismatch is one layer beyond the exit. **Next step is to instrument that layer's
backward specifically** -- a grad hook on whatever `from_local` is receiving -- rather than
adjusting the seal a seventh time. Every one of the six was a guess about placement; the
one measurement taken (the forward/backward kind asymmetry) is what made the direction
clear, and it should have come first.

The tower's TP and its dynamic CP are mechanisms this migration deliberately leaves alone,
so an acceptable resolution is also to keep the whole tower behind a plain boundary in
both directions and let the stream be DTensor only from the splice onward.

## The declarative layer actually drives, at last: 4 -> 53 modules

Three things had to be true at once, and each was hiding the next.

**1. The sweep must run AFTER the driver.** It lived at the end of `apply_tp`, which is
before `_drive_declarative_sharding`, so it promoted every declared-but-not-yet-distributed
parameter to `DTensor(Replicate)`. The driver then found them distributed with the wrong
placement and refused: "already a DTensor with placements (Replicate(),), but its
sharding_config expects (Shard(dim=0),)". Extracted to
`_sweep_remaining_to_replicate` and called after the driver, where "remaining" finally
means what the name says.

**2. `_already_distributed` must not recurse.** It asked `m.parameters()`, so a parent whose
children the imperative plan had covered counted as finished -- and `parallelize()` only
touches what a module itself declares, so the parent's own declarations were never applied.
That is how the nine KDA layers' `A_log` and `dt_bias` reached `clip_grad_norm_` as plain
tensors while everything around them was a DTensor.

**3. Three imperative plan entries had to go**: the dense FFN's Colwise/Rowwise trio and
both `shared_experts` sites. They state the same split the declarations state, and once the
driver stops skipping those modules only one side can act.

Result: the driver enters **53** modules -- 27 `AttnResProjection`, 13 `KimiMLP`, 9
`KimiDeltaAttention`, 4 `ScaledDotProductAttention` -- against 4 before. `tp2` passes and
the loss moves from 12.06776 to 12.03708, which is expected: replicated projections
becoming a real column/row split is a different floating-point order for the same
mathematics.

### Two process failures worth keeping

**Deleting an imperative entry is not handing the module over.** Three times today a
removal was followed by a green matrix and no change in what the driver entered, because
the sweep caught the parameter instead. The only reliable signal is the module count, and
54/54 cannot distinguish the two.

**Two edits were lost without any error.** One went to a path under a directory that
`rm -rf` had just removed -- the write "succeeded" into nowhere. Two others sat in commits
that a later `git reset` discarded, and I re-derived the same deletions a second time
without noticing they had once existed. In both cases the code read as if the change was
present. What caught it was measurement: the conflict probe reporting exactly the same 36
mismatches, and the driver reporting exactly the same module count.

### First real check of the declarations: text arm 6/13, and the failure is informative

With the driver actually entering 53 modules, the text arm's `tp` cells fail with

    Redistribution from one partial type (P(sum)) to another
    (MaskP(sum, torch.Size([2016, 512]), 0)) is unsupported

`[2016, 512]` is `embed_tokens`, and `MaskPartial` is what vocab-parallel embedding
produces. `embed_tokens` itself has NO declaration -- it is still driven by the imperative
`RowwiseParallel` -- so the `MaskPartial` side is the old path. The other side is new:
`--debug.detect-anomaly` puts the forward at `attn_res.py:118`, `block_attn_res_tensor`,
which is the aggregation the 27 now-declared `AttnResProjection` modules feed.

So the two mechanisms meet inside the AttnRes aggregation, one producing a plain `P(sum)`
and the other a `MaskPartial`, and DTensor has no conversion between two different partial
types.

This is what "the declarations have never been checked" looks like when it finally gets
checked: not a wrong placement, but an interaction that could not exist while the
declarations were inert. It is progress, and it is also the reason the multimodal `tp2`
passed earlier while the text `tp2` does not -- the multimodal path routes the embedding
through `_splice` rather than straight into the residual stream.

Next: decide which side changes. Either `embed_tokens` also takes a declaration so both
ends agree on the partial type, or the aggregation redistributes to `Replicate` before
combining. The first is the direction of the migration; the second is a local patch.
