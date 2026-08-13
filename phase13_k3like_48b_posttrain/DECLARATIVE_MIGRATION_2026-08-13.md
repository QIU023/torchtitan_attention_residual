# Migrating kimi_k3 to declarative sharding: the blockers are gone

## Why this is not optional any more

`torchtitan/models/kimi_k3/parallelize.py` is the **only file left in
`torchtitan/models/` that calls `parallelize_module`**. Every upstream model, including
llama3, now carries a `sharding.py` and declares layouts on its config tree;
`llama3/sharding.py` landed 2026-06-25, before our last sync, so this migration was
already finished upstream while we were not looking at it.

`use_local_output` appears **once** across all of upstream's models
(`common/rope.py`). It appears **20 times** in ours.

Technically nothing forbids the imperative path -- `sharding_config is None` makes the
redistribution machinery a no-op (`protocols/module.py:68`), which is why 11 of 13 matrix
cells still pass. The accurate framing is worse than "forbidden": we are the **sole
remaining user**, so no one has any reason to keep the imperative path working. The EP x TP
break is the first time that bill came due, and the way it came due is instructive --
upstream split one variable used for both `out_src` and `out_dst` into two expressions
keyed on different flags. Nobody could have been expected to notice that this breaks a
usage pattern that does not exist upstream.

## The three constraints that justified the imperative plan: all retired

This is the part that changes the size of the job. The plain-tensor boundary was
documented as the forced consequence of three hard constraints. Re-checked against the
code, none of them binds.

### 1. fla's triton kernels do not dispatch through DTensor -- ALREADY SOLVED, in our code

`model.py::_to_local_if_dtensor`'s own docstring:

> KDA forward stashes the DTensor mesh+placements, strips DTensor from x and from each
> weight **at the kernel call site**, runs the kernels on plain tensors (each rank computes
> redundantly under Replicate), and **re-DTensors at the end** so the parent NoParallel
> output hook composes correctly.

So KDA is already DTensor-in / DTensor-out, with the conversion pushed down to exactly
where it belongs. `NoParallel(use_local_output=True)` in the plan strips an **additional**
layer at the module boundary, which the kernels do not need.

`_patch_fla_for_dtensor` (117 lines) does the same for `ShortConvolution` and
`FusedRMSNormGated`, bound per instance rather than patched onto the fla classes. The
pattern is correct and stays.

### 2. PP's P2P can only send plain tensors -- HANDLED BY THE FRAMEWORK

`torchtitan/distributed/pipeline_parallel.py`:

> Pipeline parallelism requires an SPMD mesh during module split so that at runtime the
> current PP rank can reconstruct a DTensor after receiving a plain tensor from the previous
> PP rank. DTensors are not directly serializable across PP stages (because ProcessGroup is
> not serializable), so each stage uses this callback to obtain its local DeviceMesh and
> **re-wrap incoming tensors as DTensors with the correct placements**.

Upstream unwraps and re-wraps at the stage boundary itself. It is not a model concern, and
it is how upstream models run PP together with declarative TP.

### 3. AttnRes's `torch.stack` cannot mix plain and DTensor -- SELF-INFLICTED

The failure is MIXING. A stack of all-DTensor operands is fine. The mixture exists because
some boundaries were stripped and others were not; removing the stripping removes the
mixture.

**Conclusion.** The plain-boundary design is a historical layer left behind after the fla
problem was fixed at the right level, not a forced consequence. It was asserted as
"the only solution forced by three hard constraints" twice in one session without either
assertion being checked -- see `HOW_I_GET_THIS_WRONG_2026-08-13.md`, mechanism 5.

## What the migration is

| step | content |
|---|---|
| 1 | Write `kimi_k3/sharding.py` on the `deepseek_v3/sharding.py` template (192 lines, also MLA + MoE): declare layouts on the config tree for attention, norms, dense FFN, and hand the MoE subtree to `set_moe_sharding_config` with the REAL `enable_sp` |
| 2 | Remove `use_local_output=True` boundary-by-boundary so activations stay DTensor. `_to_local_if_dtensor` and `_patch_fla_for_dtensor` DO NOT CHANGE -- they are already at the right level |
| 3 | Delete the parts of `apply_tp_kimi_k3` (577 lines) that the declarative path replaces |
| 4 | EP x TP resolves as a consequence: with a DTensor MoE boundary the upstream SP layout has something to redistribute to |

Step 4 is the reason to do this now rather than later. It removes the need for the three
core changes to `common/moe_sharding.py` (tried, deadlocked, reverted) AND the need for an
explicit `ep>1 and tp>1` raise. Both were being planned as workarounds for a problem the
migration dissolves.

## Verification, and why "byte-identical" is not the criterion

Every step can change numerics: removing a boundary conversion changes summation order.
The criteria are therefore:

1. all expressible cells train 10/10 on all three arms;
2. losses stay same-order and same-trend against the post-merge baseline below (the 5th
   significant digit may move);
3. the two EP x TP cells go from FAIL to pass.

Three arms per step, serial: text (`kimi_k3_mini_block_attn_res`), multimodal full
(`kimi_k3_debugmodel_report_arch`), multimodal LoRA (`..._lora`). Text LoRA is deliberately
not run -- it would only tell us whether a LoRA problem is multimodal-specific, and no such
problem is open. K3 trains MoonViT-V2 jointly, so multimodal is the target and text-only is
an instrument.

## Post-merge baseline, 2026-08-13 (before any migration step)

Merged tree at upstream `65fa556be`, DEP and dynamic CP on for the multimodal arms,
steps=10, seed 42, deterministic. Tables in `/workspace/mx_postmerge/*_13.txt`.

| arm | passing | failing |
|---|---|---|
| multimodal full | 11/13 | `ep2_fsdp2_tp2_pp2`, `ep2_fsdp2_tp2_cp2` -- the EP x TP break |
| multimodal LoRA | 11/13 | same two |
| text | **4/13** | see below -- the text arm is not currently a usable instrument |

### The text arm needs fixing before it can verify anything

Two causes, neither of them the migration's concern, but both blocking its instrument:

* **4 cells (all with PP): `KIMI_VIT_DEP_STAGES=1: this rank owns 1 vision stage(s) by
  stage index but 0 were wired`.** The gate script exported `KIMI_VIT_DEP=1` for the text
  arm. DEP on a text flavor is INVALID, not inert -- already established, already cost a
  text-arm launch once. Fix: set the topology per arm rather than exporting globally, which
  finding 32's config fields now make natural (`vit_dep=False` for the text arm).
* **5 cells (no PP): `Failed to set the allowed dynamic shared memory size to 108160`.**
  fla's SM120 shared-memory ceiling -- the same class as `Raising_PRs/PR13*`. Note the
  squeeze: `seq_len 8192` (the flavor default) OOMs at 16 GB, and `seq_len 2048` trips this.
  A workable seq_len for this box has to be FOUND, not assumed, and the fact that
  `chunk_kda`'s blocking follows the input shape is why one value can OOM and another can
  exceed shared memory.

## Consequence for PR-A that must not be forgotten

`Raising_PRs/k3_pr_a_tp_kda/PR_BODY.md` argues that a declarative `sharding_config`-only
approach "does not work here" because "the declarative vocabulary has no
`use_local_output`, so it cannot express a plain-tensor boundary". After this migration
that argument is **void**, because the plain-tensor boundary is not needed. PR-A's body and
its `commits.md` extraction list both have to be rewritten, and the PR shrinks from
"577-line imperative plan" to "a `sharding.py` plus the fla kernel-boundary shims" -- which
is a far easier review and closer to what the maintainer asked for in the first place.
