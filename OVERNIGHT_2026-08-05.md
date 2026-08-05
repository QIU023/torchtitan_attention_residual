# Overnight run: anchors and task order

## Restore points (recorded before any of tonight's work)

| repo | branch | commit |
|---|---|---|
| logbook | main | ac7496b |
| fork torchtitan | attention_residual_dev | b99ff970c |
| fork verl | kimi_k3_integration | cb9ac9e1 |

PR bookmarks `k3_pr_{a,b,c,d}` are all at `a146d1bf2` and are NOT touched tonight.

To roll back anything from tonight: `git reset --hard <commit above>` in the
repo concerned. Every commit tonight is separate and revertable on its own.

## Task order (from the user)

1. ViT real TP
2. ViT dynamic CP
3. ViT DEP -- vision compute scheduled into PP bubbles
4. 32-layer multimodal PP8xVP4 stress test (completes the PP line)
5. veRL GRPO
6. full MTP support
7. LoRA TP debug

Rule for tonight: if a task blocks, try a few approaches, then move to the next
one and record why. Do not leave the tree half-edited.

## Log

### Task 1 -- ViT real TP

**Landed (fork `54810694d`): MoonViT encoder MLPs tensor-parallel.** fc0 Colwise
/ fc1 Rowwise, exact because the GELU between them is elementwise and commutes
with the shard. Verified as a function, not by the run completing: TP2 against
replicated on identical input gives `max_abs 3.9e-03, rel 5.2e-03` -- bf16
rounding. End to end that moves step-1 loss by 0.013 at tp2, consistent with a
5e-3 relative perturbation of the vision features through 13 layers. tp2, tp4
and fsdp2 x tp2 x cp2 all run; suite 260 passed.

Two facts that had to be measured rather than reasoned about, and that the next
attempt should not rediscover:

* The TP plan must run BEFORE `distribute_module` replicates the rest. The styles
  need plain-tensor parameters, and a later `distribute_module` leaves
  already-distributed ones alone.
* `fc1` must return a DTensor (`use_local_output=False`). `encode_images` lifts
  the tower activations at the boundary, so the block residual is a DTensor and
  the plain form fails the add on mixed Tensor/DTensor.

**Attention head sharding: attempted, NOT landed, reverted.** The approach that
avoids touching the checkpoint contract is to keep `wqkv` replicated (every rank
projects all heads), slice the local head range, and let `wo` be RowwiseParallel
-- `[L, A_local*K]` is exactly a `Shard(-1)` of `[L, A*K]`.

The obstacle, worth keeping: **slicing a `Replicate` DTensor gives another
`Replicate` with a smaller logical shape**, which loses the fact that it is a
shard, so `wo` sees a logical `[L, A_local*K]` against a weight whose reduction
dim is the full `[L, A*K]` and raises "a and b must have same reduction dim".
The fix is to `to_local(grad_placements=[Partial()])` before slicing -- Partial
is right by construction, since each rank's gradient is its own additive
contribution to the replicated `wqkv`.

With that fix the TP tower runs and produces finite output. It was reverted
anyway because the parity harness could not confirm the numerics: the
**reference** side came back NaN (`NAN ref: True  tp: False`), so the harness is
what is broken, not obviously the implementation -- and an unverified numerical
change should not sit in the tree. Suspect `out = torch.empty_like(q)` filled
per `cu_seqlens` segment, which leaves rows uninitialized if the segments do not
cover `L`.

Next attempt starts by fixing the harness (or comparing inside a real training
step instead of a standalone tower), not by rewriting the sharding.

Tree left at `54810694d`, suite green.

**Follow-up:** the tower is clean single-process -- same config, same input, no
TP, no DTensor: `has_nan: False`. So the NaN was specific to building two towers
inside one distributed harness, i.e. a harness artifact, and MoonViT has no
latent uninitialized-row bug. That makes the reverted attention sharding
*probably* correct, but probably is not verified, so it stays out. Resume by
validating it inside a real training step.

### Task 2 -- ViT dynamic CP: design settled, and one recorded blocker is wrong

Today, under CP, the vision tower is **replicated across CP ranks**: text tokens
are sharded but `pixel_values` is not, so every CP rank encodes the whole batch's
images and throws away the part it does not need. `_cp_group` is already attached
to the multimodal wrapper.

Our own notes have said dynamic CP "needs a core change, because sub-CP-group
formation is per-batch while `DeviceMesh` is static for the run". **That is not a
blocker for the formulation that matters here.** The work to distribute is per
IMAGE, not per sub-group: each CP rank encodes a disjoint subset of the batch's
images and the features are all-gathered over the existing, static `_cp_group`.
Dynamic *work assignment*, static group. No new mesh.

The obstacle that looks fatal and is not: per-image feature blocks have different
lengths, so the gather is uneven. But `grid_thw` is replicated -- it is part of
the batch, not sharded -- so **every rank can compute every image's output length
before the collective**: patch count is `prod(grid_thw[i])`, and the projector's
2x2 merge divides it by 4. Sizes known a priori means a fixed-shape
`all_gather_into_tensor` with precomputed offsets, not an object gather.

Sketch, in `encode_images`:

1. `counts = grid_thw.prod(-1)`, `out_len = counts // merge**2` -- both replicated.
2. Rank `r` of `cp_size` takes images `r::cp_size` (round-robin balances better
   than contiguous when image sizes vary).
3. Pack and run the tower on that subset only.
4. `all_gather_into_tensor` into a `[sum(out_len), D_llm]` buffer using the
   offsets from step 1, then split back into the per-sample list.
5. Backward: the gather's transpose is a reduce-scatter, so each rank receives
   exactly the gradient for the images it encoded. `funcol.all_gather_tensor` is
   differentiable and does this; a manual `dist.all_gather` is not.

Correctness check to run first, before wiring it into the trainer: cp2 with
image-sharded encode against cp2 with the current replicated encode, same seed,
same step-0 checkpoint. They should agree to bf16 rounding -- the tower is a
per-image function, so distributing images changes no math at all, unlike the MLP
shard which changes summation order. **That makes this the easier of the two to
verify: exact equality is the expected result, not approximate.**

Not implemented tonight -- landed with the design rather than a half-wired
forward, since the remaining budget was not enough to implement and verify it.

### Task 2 -- ViT dynamic CP: implemented, forward verified exact, backward NOT. Reverted.

The design above was implemented: round-robin image ownership over the static
`_cp_group`, output lengths sized from the replicated `grid_thw`, a fixed-shape
`funcol.all_gather_tensor`, and a `KIMI_VIT_CP_IMAGE_SHARD=0` switch so the two
paths can be A/B'd on one configuration.

**The measurement discipline mattered here twice.**

First run looked perfect: cp2 gave 12.07565, bit-identical to the published cp2
leg. It proved nothing -- the path never engaged. The debug flavor runs
`local_batch_size 1`, so one image per microbatch and `len(counts) < cp_size`
sent it straight to the fallback. Confirmed by instrumenting rather than by
inspection: `engaged=0`.

With `--training.local-batch-size 2` the path fires on both ranks
(`engaged=2`), and the A/B is:

| | step 1 loss | step 1 grad_norm | step 2 loss | step 2 grad_norm |
|---|---|---|---|---|
| sharded | 12.07338 | 12.4375 | 11.99565 | 10.1250 |
| replicated | 12.07338 | 12.4375 | 11.99084 | 10.0625 |

**Forward is bit-exact, as predicted.** Step 2 diverges, so the backward is not.

The hypothesis, and it is specific rather than a shrug: in the replicated path
every CP rank computed the tower's gradient over ALL images, so each rank already
held the complete sum and nothing needed to reduce it across CP. With images
sharded, the all-gather's transpose is a reduce-scatter, so each rank ends up
holding the gradient for only the images it owned -- an under-reduction, unless
something all-reduces tower parameter gradients along the CP axis. Under
FSDP/DP the reduction happens on the dp axis, which is not this one.

That is the same failure shape as the shared-code MoE gradient bug filed earlier
(under-reduced gradients, invisible in loss because Adam is scale-invariant, and
visible in grad_norm) -- worth checking whether it is literally the same missing
reduction.

Next step, in order: (1) establish whether torchtitan reduces gradients over the
CP axis at all for this model; (2) if not, all-reduce the tower's parameter
gradients over `_cp_group` after backward, or make the collective's backward
produce the full sum instead of the scatter; (3) re-run the same A/B and require
step 2 to match as well.

Reverted -- forward-only correctness is not correctness. Tree at `54810694d`,
suite 260 passed.

### Task 2 -- LANDED after all (`57c6728ed`). The under-reduction hypothesis was wrong.

The revert above was premature. Two measurements settled it:

**fp32 A/B.** A missing CP reduction would show as a factor-of-cp error. It does
not: step-1 grad_norm is 12.4583 vs 12.4584, i.e. ~1e-5 relative, which is
accumulation noise. Also relevant, and it undercuts the hypothesis directly:
torchtitan's `"fsdp"` mesh IS `dp_shard x cp`, and `disable_fsdp_gradient_division`
is already applied, so tower gradients are summed across the CP axis rather than
averaged. There was never a missing reduction.

**Dense control.** Step 2 differing 400x more than step 1 looked like MoE top-k
route flipping. The dense flavor, which has no routing to flip, shows the same
pattern: step 1 bit-identical (12.05145, grad_norm 9.6241), step 2 apart by
2e-4. So it is neither under-reduction nor route flipping -- it is
reduction-order difference below grad_norm's display precision, amplified by
Adam's first updates, where the update magnitude is ~lr almost independently of
gradient scale.

Final evidence for landing: step-1 loss and grad_norm bit-identical in bf16 AND
fp32, on the MoE flavor AND the dense control. By CONTRIBUTING's rule this is a
computation change judged on convergence, so step-1 identity is stronger than
required. Published matrix legs are untouched -- they run `local_batch_size 1`,
so the path never triggers.

**Lesson worth keeping:** "step 1 matches, step 2 does not" is not evidence of a
backward bug on its own. Adam makes a 1e-5 gradient difference into a 1e-4 loss
difference in one step. Distinguishing a real reduction bug from that needs a
dtype sweep (a factor error survives fp32; rounding does not) and an
architecture control, not more staring at the same two numbers.

Tree at `57c6728ed`, suite 260 passed.

### Task 1b -- ViT attention head TP: two hard prerequisites, not landed

Re-applied with an env gate and tried to A/B it the way task 2 was settled. Both
halves of that method are unavailable here:

1. **The debug vision tower has 3 attention heads.** They do not divide 2 ranks,
   so the guard correctly refuses to shard and the A/B compares nothing. Every
   multimodal flavor inherits this tower. A verification needs a vision config
   with an even head count -- MoonViT-V2 ships 12, so this is a debug-config
   artifact, not a design limit.
2. **fp32 + tp2 is hardware-blocked on this box**: `Failed to set the allowed
   dynamic shared memory size to 108160` against this card's 101376 -- the
   documented KDA fp32 ceiling. So the dtype sweep that distinguished
   reduction-order from a factor error in task 2 cannot be run for TP legs here.
   (It worked for the cp2 legs, which is why task 2 could be settled.)

The implementation itself is believed complete: `wqkv` replicated, `to_local(
grad_placements=[Partial()])` before slicing so the shard identity is not lost,
`wo` as RowwiseParallel over `Shard(-1)`. Reverted rather than shipped behind a
default-off flag, because unverifiable code behind a flag is how dead code
accumulates.

To finish it, in order: add an even-head debug vision config; A/B in bf16 with
the dense control (fp32 is unavailable for TP legs on this hardware); require
step-1 loss and grad_norm identity, as task 2 did.

### Task 1b -- LANDED (`e37faff3c`). Both blockers removed.

Added `kimi_k3_debugmodel_report_arch_vit4h` (4 heads over the same
`qkv_hidden_size` 384 -> head_dim 96, RoPE-legal, parameter count unchanged) so
head sharding can be exercised at all, and verified the sharding as a FUNCTION
instead of through end-to-end loss, since fp32 is hardware-blocked for TP legs
here.

Single encoder block, TP2 against replicated on identical input:
`max_abs 7.8e-03, rel 2.5e-03`, no NaN -- bf16 rounding, same order as the MLP
shard's 5.2e-03.

The end-to-end A/B on `vit4h` at tp2 moves step-1 loss by 1.5e-3
(12.06875 sharded vs 12.06723 replicated), which is consistent with a 2.5e-3
relative perturbation of the attention output and is NOT evidence either way at
bf16 precision -- which is exactly why the module-level measurement was needed.

`KIMI_VIT_TP_HEADS=0` keeps the A/B available.

## Policy recorded

**The full parallelism matrix runs MULTIMODAL only from now on.** The dense and
text-only legs were controls for specific questions (is the run-horizon band
architecture-specific; is a disagreement route flipping) and stay available as
controls, but they are not part of the reported matrix.
