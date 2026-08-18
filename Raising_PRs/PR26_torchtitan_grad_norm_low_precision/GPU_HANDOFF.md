# GPU-box handoff: finish verifying the pytorch `get_total_norm` dtype change

Everything below was prepared on the Windows logbook box, which is CPU-only with torch
2.8. The change is written and the CPU half is measured. **What is left is everything that
needs a GPU, a DTensor, or a current torch** -- and that is the half the change exists for.

## What this is

`torch.nn.utils.get_total_norm` accumulates in the input tensors' dtype. With bf16 that
makes the total depend on how the tensors were GROUPED, and under PP or EP the grouping is
where the model was cut -- so two layouts of one model with bit-identical gradients report
different `grad_norm` and clip differently. torchtitan carries a private fp32 copy of the
function for this (the grad-norm PR, upstream 4135). A pytorch core maintainer invited an
issue/PR to fix it at the source, which is this.

## State, and the line between measured and read

**Measured** (Windows, torch 2.8.0+cpu, plain CPU tensors):

| | result |
|---|---|
| patch applies to `pytorch/main` | `git apply --check` clean, result byte-identical to the intended file |
| `torch._foreach_norm(x, 2, dtype=None)` | accepted, returns bf16 -- the unconditional passthrough is fine |
| grouping spread, bf16 | 2.490e-03 relative |
| grouping spread, fp32 | 1.004e-07 relative |
| returned dtype, default unchanged, empty case, `foreach=False` | all assert clean |

**Read, not measured -- this is your list:**

* **DTensor.** The entire motivation. The PR body claims "passing `dtype` preserves the
  `_NormPartial` placement rather than materializing anything". That is read off the source,
  never run. If it is wrong the PR body is wrong.
* **CUDA.** Every number above is CPU. `_foreach_norm`'s dtype path on CUDA is untested here.
* **`foreach=True` explicitly**, and mixed-dtype inputs (bf16 params next to fp32 ones).
  `_group_tensors_by_device_and_dtype` groups by dtype, so `norms` is heterogeneous today
  and homogeneous with `dtype` set; nothing checks what that does.
* **Where pytorch tests `get_total_norm`.** Not located. The probe is our harness, not a
  drop-in for theirs, and a PR without a test in the right file will bounce.

## Rebuild the branch (it exists only on the Windows box, unpushed)

The patch is committed in the logbook, so this needs nothing from that box:

    cd <logbook> && git pull
    git clone --filter=blob:none --depth 1 --sparse https://github.com/pytorch/pytorch.git pytorch_pr
    cd pytorch_pr
    git config core.autocrlf false
    git sparse-checkout set torch/nn/utils
    git checkout -b get-total-norm-dtype
    git apply <logbook>/Raising_PRs/PR26_torchtitan_grad_norm_low_precision/get_total_norm_dtype_pytorch.patch

The blobless sparse clone is 4 MB and still produces a real pushable branch -- the index
carries the full tree, so the commit is complete even though only `torch/nn/utils` is
checked out. If the patch does not apply, `main` moved; regenerate rather than hand-edit,
and say so in the report.

`core.autocrlf false` is not optional on a Windows checkout: without it every sparse file
comes out CRLF and shows as a full rewrite in `git diff`. On Linux it is a no-op.

Note the branch commit carries a `Co-Authored-By: Claude` trailer. Keep or drop it, but it
is a human's decision, not something to change silently.

## Run the probe

    python Raising_PRs/PR26_.../probe_get_total_norm_dtype.py
    python Raising_PRs/PR26_.../probe_get_total_norm_dtype.py --module <pytorch_pr>/torch/nn/utils/clip_grad.py

`--module` loads one file and uses its `_get_total_norm`, so the patched code runs without
touching the installed torch. First arm should show the bf16 spread, second should print
`all passed`.

## Two traps this already cost, do not pay them again

**A probe can refute a real defect.** The first version of this one stacked the bf16
partials and normed them in bf16. At magnitude ~158 the bf16 grid spacing is 2.0, so all
four groupings snapped to 158.000000 and the table read as if nothing was wrong. The groups
are RANKS and the cross-rank combine is a separate step, so the partials have to be combined
in higher precision or the per-group rounding has nowhere to show. The probe now combines in
float64 and says so in its docstring.

**The patched assertion is a tolerance, and has to be.** fp32 grouping differences shrink,
they do not vanish -- each group's norm is still rounded, at 2^-23 instead of 2^-8. A
bitwise judge would fail a correct patch. If you tighten it, tighten it to a number you
measured.

## The DTensor check, concretely

The cheapest form is 2 ranks, gloo, no model:

* build a list of DTensors on a 1-D mesh, some `Shard(0)` and some `Replicate()`;
* call `_get_total_norm` from the patched module with and without `dtype=torch.float32`;
* assert it returns without a mixed-dispatch error, and print the result's type and
  placements at each step -- specifically whether the per-tensor `vector_norm` output is
  still a DTensor with `_NormPartial`, since that is the claim in the PR body;
* compare the value against `full_tensor()`-ing everything first and norming on one rank.

If `_NormPartial` is NOT preserved, the fix is still right but the PR body's sentence about
it must go, and torchtitan's own copy needs re-examining for the same reason.

## Then, if it holds

1. Report the numbers back to the logbook (a file in this folder is fine).
2. The `ISSUE_pytorch.md` and `PR_pytorch.md` PASTE blocks are ready; **neither has been
   posted and neither should be without a human's go-ahead.**
3. Optional and useful: re-run the torchtitan-side repro from `PR.md` on the current torch
   (llama3_debugmodel, 4 GPUs, dp_shard=2 x pp=2, bf16, patch on/off) so the downstream
   table is not stale when the pytorch discussion asks what it fixes.

## What NOT to do

* Do not push the branch or file anything without being asked.
* Do not put `owner/repo#N` or a github URL for a third-party issue in any COMMIT message
  (CLAUDE.md). Links belong in the PR/issue text we deliberately post.
* Do not make the torchtitan PR wait on this one -- the reasoning is in `PR_pytorch.md`, and
  it is a decision already taken.
