# Handoff, 2026-08-08 evening

Supersedes `HANDOFF_2026-08-08.md`, which covers the same day up to the DEP stage
split. Everything after that -- the bubble negative result, the side stream, the
run-ahead, and the veRL sync verification -- is only here.

**Heads:** fork `QIU023/torchtitan` `attention_residual_dev` at `d60f120a6`;
`QIU023/verl` `kimi_k3_integration` at `a94e0fe3`; logbook `main` at `a574507`.
kimi_k3 suite: **293 passed**, 1 skipped. Disk watchdog running as a supervisor
service.

> **UPDATE, later the same night.** The run-ahead below is FIXED and engaging, and the
> pp8xvp4 latency A/B has been run. See "Run-ahead: fixed" immediately below; the
> original entry is kept after it because its reasoning was wrong in an instructive
> way. Full detail in `phase13_k3like_48b_posttrain/VIT_DEP_DESIGN_2026-08-07.md`
> (read bottom-up) and `PP_ADAPTER_UNDER_SIDESTREAM_VIT_2026-08-08.md`.

---

## Start here: run-ahead is FIXED; the latency A/B is not decidable on this box

**Engaging.** Two bugs, neither the one predicted below:

1. `_install_vision_prefetch` was called only from `pipeline_llm_with_cache_adapter`,
   while every kimi_k3 flavor registers `pipeline_kimi_k3_with_cache_adapter`. Dead
   code -- including the diagnostic that was supposed to reveal it.
2. `logger` was never imported and the only line using it was that dead install log,
   so adding the call surfaced a latent `NameError` on rank 0.

**Engagement is asserted from a count predicted before measuring**, not from loss:
pp2 with 4 micro-batches and depth 1 must give mb0 miss + mb1..3 hit. Predicted 3/1,
measured 3/1. pp8xvp4 with 32 micro-batches: predicted 31/1, measured 31/1. Loss and
grad_norm bit-identical to the off arm (12.07786/12.5625, 12.04995/8.3125).

**The latency A/B was run as asked (pp8xvp4 multimodal, total latency, DEP toggled by
one parameter, plus the negative control) and does NOT resolve the effect:**

| arm | ms/step | repeats |
|---|---|---|
| base | 2063.3 | -- |
| dep_nopf (negative control) | 2095.6 | 2102.9 / 2059.8 / 2101.5 |
| dep_pf | 2072.6 | 2088.3 / 2121.2 / 2063.5 |

Medians -0.6%; within-arm spread 2.1% and 2.8%; ranges overlap; **the sign flips
between repeats.** Two reasons it could not have resolved it, both knowable in
advance: mfu is 0.12% in every arm (so ~2.5ms of a 2063ms step is compute, capping any
win at ~0.12%), and the theoretically hideable share AT THIS FLAVOR IS ZERO -- so a
correct implementation must also show nothing here.

**`phase13_k3like_48b_posttrain/dep_hiding_theory.py`** is the hardware-independent
answer: 56% hideable at the real `2p8t_vl` cost ratio (limited by TIMING, not
capacity), 0% at the debug flavor. Run it with `--sweep`.

**Also found:** the cross-stage cache adapter **never engages under the multimodal
wrapper** -- both ranks fall back, each for a different reason. Numerically safe
(symmetric, and loss is bit-identical with the flag on and off) but zero coverage.
The matrix never sets the flag, so matrix results stand; the adapter's PP8xVP4
memory/tps numbers remain valid as a TEXT-only claim. Two separate defects to fix if
it is wanted there: the wrapper class exposes no `num_blocks`/`layers_per_block`, and
the layout-table inference reads a layer count that does not match the split.

---

## The original entry, kept because its reasoning was wrong instructively

**The DEP run-ahead does not engage.** `KIMI_VIT_PREFETCH=1` at pp2 gives loss and
grad_norm identical to the off arm (12.07786 / 12.5625) **and the
`prefetch installed` log line count is ZERO in both arms**. So the identical numbers
are the signature of a path that never ran, not evidence about the prefetch.

Narrowed, not solved:

* the registered `pipelining_fn` is correct (`pipeline_kimi_k3_with_cache_adapter`),
* the DEP vision stage IS built -- stage 0 carries the `__kimi_dep_vision__0` FQN,
* so `_install_vision_prefetch` runs and either finds no `KimiK3ViTStage` in
  `model_parts` or returns early.

**Next diagnostic, concretely:** log `[type(m).__name__ for m in model_parts]` inside
`_install_vision_prefetch` (`pipeline_adapter.py`). First suspect is the `isinstance`
against the imported class, because what lands in `model_parts` is
`parallelize_fn`'s RETURN value, and under DEP that return goes through
`inner_parallelize`.

Then, and only in this order: **log line present → numerics exact → latency A/B with a
negative control.**

**What was wrong with it:** the third bullet asserts "`_install_vision_prefetch`
runs", and it did not -- it was never called from the registered `pipelining_fn`. The
guessed `isinstance` culprit was downstream of a function that never executed, so the
proposed diagnostic could not have printed anything either. The lesson is narrow and
worth keeping: **before debugging what a function does, establish that it is
called.**

---

## DEP: the decoupling works; the bubble hiding does not, and that is proven

### What is implemented and exact

ViT and text as separate PP stages, `KIMI_VIT_DEP=1`, default off because it changes
the stage count. From a **shared seed checkpoint** -- mandatory, since different stage
splits consume RNG differently and cold start gave a misleading 0.05 --

| configuration | loss | grad_norm |
|---|---|---|
| pp2, 1F1B | identical to non-DEP (12.05471) | 1 bf16 step apart |
| pp4, 1F1B | **identical** (12.07418) | **identical** (12.5000) |
| pp2, Interleaved1F1B | **identical** (12.07418) | **identical** (12.50) |

So moving the tower onto its own stage changes no forward arithmetic, and the tower's
placement is now configuration rather than a hard attachment to the embed stage.

### What is proven NOT to work, three ways

1. **Bubble count is unchanged.** pp8 x Interleaved1F1B, 16 virtual stages, torch's
   own dependency simulator: **2019 bubbles with DEP, 2019 without.** A stage's
   actions are on the critical path, and since the vision stage is taken out of the
   text budget, DEP swaps a text stage for a vision stage -- same dependency graph,
   same stalls, vision work in a COMPUTE slot rather than a bubble.
   Probe: `matrix_scripts/../dep_bubble_structure.py`.
2. **Every IR reorder is rejected.** 24 candidates (`keep_first` x `lookahead`), action
   multiset verified preserved each time, all refused by the lowering with
   `Malformed compute schedule, can't schedule sends/recvs`. Deferring a vision
   forward on one rank leaves the consumer's forward earlier in ITS list, so no
   send/recv pairing exists. `dep_reorder.py` is kept as the record of that.
3. **The reason both hold:** at one vision stage the ViT IS the pipeline head, so the
   bubbles are downstream of the work that would fill them.

Requirement (2) is quantified as entirely unmet, not partially: **64 vision slots on
rank 0, zero on the other seven.**

### The corrected reading of 5.2.3

Two of my readings were wrong and are corrected in
`phase13_k3like_48b_posttrain/VIT_DEP_DESIGN_2026-08-07.md` (read it bottom-up; the
later sections correct the earlier ones).

* "further decompose the ViT computation" is **per MICRO-BATCH**, not per depth. The
  `n_vit > 1` depth-split plan was solving the wrong problem and is retired.
* The observation about interleaved 1F1B names a **constraint**: the first
  micro-batches' text forwards happen at once so their ViT forwards cannot be
  deferred; later ones have slack.

So the mechanism is concurrency -- micro-batch m's encode overlapping m-k's text
compute on a side stream -- not placement in the dependency chain.

### And it closes a loop I opened myself

`VIT_DEP_DESIGN_2026-08-06.md` argued exactly point (3) -- "the idle that looks free is
upstream of the work that would fill it" -- and I superseded it for being "derived
from my own timing analysis rather than the report". It was right. **The lesson is
about the supersede: "this came from my own reasoning rather than the source" is a
reason to CHECK a claim, not to discard it.** Discarding it cost the whole reorder
round.

---

## The PP adapter needs no code change, but three properties matter

Full analysis: `phase13_k3like_48b_posttrain/PP_ADAPTER_UNDER_SIDESTREAM_VIT_2026-08-08.md`.

1. **"PP owns all NCCL"** buys ordering consistency on the PP communicator. A separate
   PG preserves it, and the vision collectives are **already** on a non-PP
   communicator (`cp_group = parallel_dims.get_mesh("cp").get_group()`), so that
   requirement is structural rather than new work. It does NOT rule out a cyclic wait
   across two communicators; the avoidance rule is the codebase's own: **gate a
   collective on the MESH, never on the data.**
2. **The mb index is `threading.local()` and violating it fails SILENTLY.**
   `CrossStageCacheAdapter.forward` reads a missing index as "this call is PP's shape
   inference" and diverts without raising. So a worker thread would return
   shape-inference output with no error. **Concurrency must come from a CUDA stream,
   never a worker thread.** This is the constraint the re-read was worth doing for.
3. **The adapter touches no streams at all** (grep confirms), so cross-stream tensor
   lifetime and the FSDP2 interaction are new and unowned.

---

## The side stream IS verified, through pp8 x vp4

`KIMI_VIT_SIDE_STREAM` on against off:

| configuration | loss | grad_norm |
|---|---|---|
| single GPU | 12.03482 both | 15.7500 both |
| fsdp2 | 12.05179 both | 15.4375 both |
| cp2 + dynamic CP | 12.05231 both | 15.4375 both |
| **pp8 x vp4 (32 stages), multimodal** | **12.04995 both** | **8.3125 both** |

The last row is the one that could have failed: at 32 virtual stages FSDP2's tower
all-gather and dynamic CP's collectives are both issued on the side stream while the
PP schedule is mid-flight across eight ranks. Exactness there is **necessary and not
sufficient** -- an ordering bug can be latent.

Three synchronisation edges in `_run_on_vision_stream`, all required: side waits for
current; every INPUT `record_stream`'d on the side stream or the allocator can reuse
its memory mid-read (correctness, not speed); current waits for side with every OUTPUT
marked back.

---

## Dynamic CP: complete on both clauses of 5.2.3

Verified exact in fp32 at two ranks on (1,8,4), (1,6,4), the report-arch config's own
(1,16,16) and the frame-spanning (2,8,4): `max|delta| = 0.000e+00`.

Sub-group path exercised, not just unit-tested:

| configuration | layout | loss |
|---|---|---|
| 2 images, threshold 256 | 1 sub-group of 4 | 12.03832 |
| 2 images, threshold 16 | **2 sub-groups of 2** | **12.03832** |
| 3 images, threshold 16 | 2 sub-groups, **2/1 split** (empty pass runs) | 12.04008 |
| 4 images, threshold 16 | (4,1) -> g=1, declines to round-robin | no engagement log |

Two layouts, same loss. The 2/1 split is where a deadlock would hide and it completed.
The last row is confirmed by the ABSENCE of the log line.

Four defects found on the way, all by measuring. The root one: **`tpool_patch_merger`
is `mean(dim=0)` over ALL frames**, so splitting a video by FRAMES is wrong at the root
(100% mismatch at t=2), and the partition must be a band of spatial ROWS with every
frame on every rank. Corollary: a partitioned image emits `(h/kh)*(w/kw)` tokens with
no `t`, so `counts // merge` is right only for stills.

Still ours rather than the report's: the `min_patches` threshold and the sub-group
count rule. The report gives neither.

---

## Cost weights cannot come from a model config

Computed from the configs, one ViT forward on a 1024-patch image is **25.2 text
layers** at `report_arch_pp8vp4` but **0.057 of one layer** at the real `2p8t_vl` --
~440x apart, because the debug flavor's text side is tiny. So any "share hidden" claim
needs the **visual token budget per micro-batch**, which is a data parameter, and
attention is quadratic in patches so large images do not extrapolate linearly.

Consequence for reporting: "DEP reduced step latency by X% at pp8xvp4 on 8x5060Ti /
PCIe" is reportable; "DEP hides most of the ViT computation" is not reportable from
that number. Both sentences have to travel together.

---

## veRL

### Verified

* **The rollout weight sync transports changing weights.** Direct, by checksum:
  live ships four distinct parameter checksums across four syncs; the
  `KIMI_GRPO_FREEZE_SYNC=1` control ships one constant value whose first equals live's
  first, so the control works and both arms provably start identically. This closes
  the handoff's open item.
* **Text GRPO runs 3 steps and learns** with a variance reward: `grad_norm` 23.6 /
  24.4 / 24.1, `rewards/mean` ~0.5, exit 0, memory stable (2.16 GB device, 34.5 GB
  host).
* Cross-engine agreement: **vLLM and torchtitan agree on logprobs to ~2e-03 in bf16.**

### Two readings that were wrong

* **`rollout_probs_diff` is NOT a sync metric**, and I framed it as one twice. It
  compares the rollout's logprobs against the training engine's recomputation for the
  same tokens under the weights the rollout used -- an engine-agreement metric by
  construction. With the freeze proven effective and the weights demonstrably
  changing, it sits at 1-2.5e-03 either way.
* **`pg_loss` is not a GRPO health signal.** It is `0.0` while `grad_norm` is 23.6, and
  that is expected: group normalisation makes advantages zero-mean and the first inner
  step has ratio 1, so `pg_loss = -mean(A) = 0` by construction while its gradient is
  large. Reading pg_loss alone gives "still not learning".

### The dependency the handoff had listed as three independent items

"multi-step stability", "a task with non-zero reward" and "weight sync measured
in-loop" are not independent: **the third is unreachable without the second**, because
with all-equal rewards `grad_norm` is exactly 0.0 and a working sync is
indistinguishable from a no-op. And merely non-zero is not enough -- a constant reward
still normalises to zero advantage. It has to VARY within the group.
`grpo_variance_reward.py` does that via a hash of the response text: deterministic (so
a differential is interpretable) and explicitly not a quality signal.

One repo hygiene fix: a modification the handoff listed as landed was sitting
**uncommitted** in the verl working tree (the tolerant processor `patch_size` read).
Committed. Also, the submodule pointer was four commits BEHIND, not diverged -- I first
read it as a divergence because I tested ancestry in only one direction, and a
one-directional test cannot distinguish "behind" from "diverged".

---

## Open, in priority order

1. ~~Make the run-ahead engage, then the latency A/B at pp8 x vp4 with the negative
   control.~~ **DONE** -- see the top of this file. What replaces it is NOT another
   latency run on this box (that question is closed as undecidable here); it is
   deciding whether the DEP work is presented as the decoupling plus a
   hardware-independent hideable-share analysis, which is what there is evidence for.
2. **Publish the corrected 18-cell matrix.** Seven TP-bearing cells changed against the
   published table and eleven are bit-identical; the `k_rot` fix accounts for NONE of
   the step-1 changes (it is backward-only, so its ablation gives identical step-1
   values) -- those come from ViT TP landing after the published matrix. Numbers in
   `MATRIX_CORRECTNESS_2026-08-07.md`.
3. **Multimodal MTP**: depth-k targets from the spliced sequence (report 3.3 puts
   visual and textual tokens in one next-token objective).
4. **Multimodal GRPO**: blocked on the debug model's MLA prefill kernels; needs
   real dims, an image-bearing RL dataset and a reward.
5. **File the `_grouped_mm` limitation** upstream.

---

## Method lessons that cost real time today

* **A tolerance loose enough to pass either way is not a measurement.** `rtol=2e-3` in
  fp32 passed with a real 1e-3 defect present.
* **Engagement must be asserted from a log, never inferred.** It caught the run-ahead
  not installing today, and the CP image-sharding path never triggering earlier.
* **A metric that reads the same whether or not the mechanism ran proves nothing** --
  `rollout_probs_diff` at `grad_norm = 0`, twice.
* **Do not use a sampling-dependent metric to test a deterministic mechanism.** The
  frozen-sync differential was uncontrolled: the two arms sampled different responses,
  so the frozen arm coming out LOWER was noise, not evidence.
* **"This came from my own reasoning rather than the source" is a reason to check, not
  to discard.** That supersede cost the entire reorder round.
* **`--training.dtype float32` does not reach the vision tower**, because CP alone puts
  FSDP in the path and FSDP all-gathers a bf16 compute copy. Every tower A/B has a one-
  step bf16 floor.
* **`n_microbatches` equals `local_batch_size`**, not `global / local`. pp4 needs
  `local_batch_size >= 4`; three failed runs came from getting this wrong, and each
  took the control arm down with it -- which is the signal that the fault is in the
  harness rather than in what is being measured.
* **`/tmp` and `/workspace` are the same overlay here.**

---

## Added to the method lessons (same night)

* **Before debugging what a function does, establish that it is CALLED.** The
  run-ahead hunt spent a round on an `isinstance` inside a function that was never
  invoked, and the diagnostic proposed to settle it lived in that same function.
* **Log success, not only failure.** Both today's silent features -- the vision
  prefetch and the cross-stage cache adapter -- logged every fallback and nothing on
  the happy path. Combined with numerical neutrality that makes engagement
  unobservable: loss cannot distinguish it and no line confirms it.
* **A rank-blind diagnostic cries wolf.** The first absence-warning fired on every
  text-only rank, i.e. on correct runs, because the vision stage is global stage 0 and
  only one rank holds it. Gate a per-rank warning on what that rank was supposed to
  have.
* **Predict the expected value from the CONFIGURATION before measuring.** "3 hits, 1
  miss at pp2" and "31/1 at pp8xvp4" were derivable from micro-batch count and depth;
  matching them is evidence, whereas a number that merely looks plausible is not.
* **Decide whether a measurement CAN resolve the effect before running it.** mfu 0.12%
  put the ceiling at 0.12% while run-to-run spread was 2-3%. That ratio was available
  from any earlier log, and it makes six 21-step runs unnecessary in hindsight.
* **A pass/fail oracle on a partial quantity always answers fail.** The first hiding
  model demanded every encode be bubble-covered and rejected a schedule matching the
  report exactly -- the report itself runs the first micro-batches synchronously.
* **`--first`/parser bugs are silent when they return a number and loud when they
  return NA.** The latency parser matched zero lines because torchtitan colourises
  metric values with ANSI codes; it printed NA, which is why it was caught. A parser
  that had silently matched the wrong field would have produced a plausible table.
* **Check the cwd before appending to a doc.** One block of this night's notes landed
  in a new untracked file inside the fork instead of the existing doc in the logbook,
  because the shell had moved. Moved back, fork tree clean.

---

## The test baseline, precisely (so the next session is not misled)

"Suite 293 passed" in this and earlier handoffs means **the kimi_k3 suite**, not
torchtitan's core unit tests. They are different commands with different baselines:

```bash
cd torchtitan && source /venv/main/bin/activate && export PYTHONPATH=$PWD
python -m pytest torchtitan/models/kimi_k3/tests/ -q      # 293 passed, 1 skipped
python -m pytest tests/unit_tests/ -q                     # see below
```

The core suite is NOT clean on this box, and none of it is ours:

* **16 failed, 533 passed** -- all 16 in `test_helion_rope.py`, cause
  `ImportError: HelionComplexRoPE override is active but 'helion' is not installed`.
  Environment, not code.
* **2 collection errors** (`test_download_hf_assets.py`, `test_tokenizer.py`) from
  `from scripts.download_hf_assets import ...` while `torchtitan/scripts/` has no
  `__init__.py`. Ignore those two files to get a runnable suite.

So the gate for kimi_k3 work is the first command. Quoting a core-suite number as if
it were the project's gate would read as a regression that is not there.
