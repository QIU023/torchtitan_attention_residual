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
