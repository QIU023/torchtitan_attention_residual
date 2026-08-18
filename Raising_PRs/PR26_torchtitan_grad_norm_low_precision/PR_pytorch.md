# pytorch/pytorch PR: `dtype` on `get_total_norm`

**Target**: `pytorch/pytorch`, `torch/nn/utils/clip_grad.py` (one function, 13 lines)
**Tag**: @janeyx99
**Patch**: `get_total_norm_dtype_pytorch.patch` -- against `pytorch/main`, `git apply
--check` clean, `py_compile` clean
**Branch**: `get-total-norm-dtype` at `3846c1e`, base `0a9cc3c`, built on the Windows box
and **unpushed** (no push credentials there; see `GPU_HANDOFF.md`)
**Evidence**: `probe_combine_precision.py` (the body's table), `probe_default_unchanged.py`
(the 144-case bitwise check, and it re-runs the body's snippet so the two cannot drift),
`probe_get_total_norm_dtype.py` (the assertions)
**Risk**: none at `dtype=None`, which is every existing call.
**Do NOT file without a human's go-ahead.**

## Audit 2026-08-18: this body was too long, and why it got that way

The first draft of the PASTE block below carried two extra paragraphs -- one on DTensor
and `_NormPartial`, one on why the argument is not added to `clip_grad_norm_`.
`PR_WRITING_RULES.md` names both: "pre-empting objections. Expect the question X.
Answer: ... Wait for the question." Nobody asked either one on the pytorch side.

How it grew, since the mechanism matters more than the edit:

* **janeyx99's two design questions were asked on the torchtitan thread, and I answered
  them in the pytorch PR body.** They belong in the reply to her and, if a separate issue
  is filed at all, in that issue. A PR description is not where a conversation from
  another repo gets replayed.
* **Writing the `_NormPartial` sentence into the body created work.** It is a claim, so it
  needed verifying, so it became a GPU task. The verification was worth having -- it
  confirmed the change composes with DTensor, which is why torchtitan can adopt it -- but
  it was CAUSED by a sentence that should not have been in the body.

Not self-inflicted, and worth the cost: the combine-precision work. It caught my own probe
reproducing nothing, and then a wrong conclusion from the GPU verification that was
proposed FOR the PR body ("cannot be reproduced on CPU"). Both would have reached a
reviewer.

**The kit can be long; the PR cannot.** That distinction is already in the rules -- the
long version lives in `commits.md` / `FILING.md` / `VERIFY_RESULTS_2026-08-18.md`, which
are for us.

**Recommendation: file ONE PR, not an issue plus a PR.** For a 13-line change with the
maintainer already inviting it, an issue is process for its own sake. Answer her two
design questions in a reply on the torchtitan thread, where she asked them.
`ISSUE_pytorch.md` stays in the kit in case she asks for the design to be split out.

Table numbers are from torch 2.14.0.dev+cu130, not the 2.8 build the first draft used:
the same seed gives different values there because `randn`'s bf16 stream changed, so a
maintainer running the block would not have matched the table.

## Verified before drafting (kit-internal, not PR content)

| claim | source |
|---|---|
| public symbol is an alias | `torch/nn/utils/__init__.py`: `_get_total_norm as get_total_norm` |
| three norm call sites | `_get_total_norm` body on `main` |
| `clip_grad_norm_` also scales | it calls `_clip_grads_with_norm_` -> `torch._foreach_mul_` |
| `vector_norm` already documents `dtype` this way | "cast to :attr:`dtype` prior to doing the accumulation" |
| ~~DTensor already takes the non-foreach branch~~ **WRONG** | I read the module-level literal `_foreach_supported_types = [torch.Tensor]` and did not check whether anything mutates it. Importing `torch.distributed.tensor` **appends DTensor to that list at import time** (measured: `[Tensor]` before, `[Tensor, DTensor]` after), so DTensors DO take the foreach path -- and `torch._foreach_norm` has no DTensor dispatch for the dtype overload, so the patch crashed on the exact input this PR targets |
| `_foreach_norm` accepts explicit `dtype=None` | measured, torch 2.8 |
| `_NormPartial` preserved under `dtype` | measured on gloo AND nccl, torch 2.14 -- `VERIFY_RESULTS_2026-08-18.md` |
| existing test | `test/test_nn.py::test_clip_grad_norm`, line ~13381, "decomposed APIs" block |

## Title

    get_total_norm: add a dtype argument for the accumulation

## Body

--- PASTE BEGIN ---

`get_total_norm` accumulates in the input tensors' dtype. With bf16 gradients split across ranks, each rank's partial norm is rounded before the partials are combined, so under pipeline parallelism the reported total depends on how many stages the model was cut into rather than only on the gradients.

The change passes a `dtype` through to the three norm calls the function already makes. One extra condition is needed: when `dtype` is set, a group containing DTensors has to take the `vector_norm` path, because `torch._foreach_norm` has no DTensor dispatch rule for the dtype overload and mis-parses `dtype` as a dim. That is the ordinary FSDP case, since FSDP gradients are DTensors.

512 bf16 gradients, the same tensors at every world size, nccl, torch 2.14. Two splits that both occur in training: FSDP, where each gradient is a DTensor `Shard(0)` and the partials combine through `_NormPartial`; and pipeline, where rank `r` owns `grads[r::world]` and the partials are all-reduced.

```
                world=1    world=2    world=4    world=8    spread(rel)
dtensor today   256.000    256.000    256.000    256.000    0.00e+00
dtensor fp32    255.682    255.682    255.682    255.682    0.00e+00
pp today        256.000    255.973    255.505    255.631    1.93e-03
pp fp32         255.682    255.682    255.682    255.682    5.97e-08
float64 truth   255.682226
```

The pipeline split varies with the world size, so the same gradients report a different norm depending only on where the model was cut, and under clipping take different-sized steps. The DTensor split does not vary but sits 0.12% off the truth. `dtype=torch.float32` puts both on it.

```
python repro_get_total_norm_dtype.py --module <clip_grad.py with this change>
```>>>>>>> f619302 (Raising_PRs/PR26: the DTensor reading was wrong, and the body follows the patch again)

https://github.com/QIU023/torchtitan_attention_residual/blob/REPLACE_SHA/Raising_PRs/PR26_torchtitan_grad_norm_low_precision/repro_get_total_norm_dtype.py

`dtype=None` is bitwise identical to today in 144 cases: 4 shapes including empty, x {bf16, fp16, fp32, fp64}, x foreach {None, True, False}, x p in {1, 2, inf}.

--- PASTE END ---

## Held back until asked

Each of these was in an earlier draft of the body. They are answers, not omissions -- post
them if the question comes.

* **DTensor.** `dtype` preserves the `_NormPartial` placement, so the cross-rank combine
  still uses the norm rule rather than a sum; measured on gloo and nccl, end-to-end
  relative error 0 on CPU and 1.06e-07 on CUDA. DTensor DOES need a special case -- see the
  guard in the body; the earlier claim that it did not was the wrong reading corrected
  above.
* **Why not on `clip_grad_norm_`.** That also scales the gradients through
  `_clip_grads_with_norm_`, so "which ops use this dtype" would be a real question there.
  A caller wanting an fp32 norm composes `get_total_norm(..., dtype=torch.float32)` with
  `clip_grads_with_norm_(...)`.
* **Why not cast the gradients instead.** That doubles gradient memory for the duration of
  the clip and changes what gets clipped; only the accumulator needs the range.
* **Cost.** The per-tensor norms are already memory-bound reads of the gradients;
  promoting the accumulator does not change read volume. The norm-of-norms is over one
  scalar per tensor.
* **`foreach=True` and mixed-dtype inputs.** Both fine; the fp32 result is invariant
  across `foreach` None/True/False on mixed bf16+fp32 input, which is what shows `dtype`
  reaches both branches.
* **How to reproduce it.** Combine the partials at higher precision than bf16. Combined AT
  bf16 the effect is quantized to the grid and can vanish for a given fixture.
* **Where it came from.** Two torchtitan pipeline layouts with bit-identical gradients
  (788 shard norms, max relative difference 0.000e+00) reported grad_norm 10.008054 and
  9.951641 against a true 9.989287; at `max_norm=1.0` clipping fires every step, so the
  two layouts took steps differing by 0.566%.

## The change

+14/-3 in one function: the keyword, a docstring entry, and four passthroughs
(`torch.tensor(0.0)`, the foreach branch, the per-tensor branch, the norm-of-norms). The
per-tensor branch reflows to four lines because the one-line form passes 88 characters
with the kwarg.

`torch.tensor(0.0, dtype=None)` is float32, so the empty case is unchanged at the default --
which is one of the 144 cases, not an argument.

The body's snippet is executed by `probe_default_unchanged.py`, so the numbers quoted in the
body are the numbers a reviewer gets by pasting it. That is worth keeping wired: the first
version of the body quoted a 394-tensor fixture while the commit message quoted a different
one, and nothing would have caught it.

## The test

Goes in `test/test_nn.py`, next to `test_clip_grad_norm` (~line 13381), which already
exercises `get_total_norm` in its "decomposed APIs" block and is parametrized over
`norm_type` / `foreach` / `device`.

Assert partition-independence, not accuracy: a test that only checks "fp32 is closer"
passes on any change that improves precision while leaving the grouping dependence, which
is the actual defect. And assert a tolerance, not bit equality -- fp32 grouping
differences shrink rather than vanish, so a bitwise judge fails a correct patch.

`probe_combine_precision.py` and `probe_get_total_norm_dtype.py` are our harnesses and
generate the numbers above; neither is a drop-in for pytorch's suite.

## What this does to the torchtitan diff

Today: `_get_total_norm_fp32`, +54/-3, plus three call-site replacements.
After: the helper is deleted and the three call sites gain one kwarg.

    -   total_norm = _get_total_norm_fp32(
    -       grads, norm_type, error_if_nonfinite, foreach
    +   total_norm = torch.nn.utils.get_total_norm(
    +       grads, norm_type, error_if_nonfinite, foreach, dtype=torch.float32
        )

x3 -- one in `clip_grad_norm_`, two in `_clip_grad_norm_with_ep`. It is a deletion rather
than a rewrite because upstream already routes DTensors down the per-tensor branch, so our
copy's explicit `isinstance(t, DTensor)` check reproduces what core does anyway.

**Do not make the torchtitan PR wait for this.** The torchtitan one is a correctness bug
that exists today and is already under review; this is an API addition that has to survive
a naming discussion. Land the helper, land this, then delete the helper in three lines.
Step 3's gate is torchtitan's minimum torch, and its from-source install already requires a
PyTorch nightly, so that is about one nightly after this merges.

The one case for the other order is a torchtitan maintainer preferring not to carry a copy
of a torch function at all. That is theirs to take, and the question belongs on the
torchtitan thread.

## An independent cross-check on the table

An earlier draft of the body carried an inline snippet: single process, CPU, each group's
norm from `get_total_norm` and the partials combined in float64. It was replaced by the
cross-rank repro, but it was run first, and what it produced is worth keeping:

    torch 2.14.0.dev20260802+cu130, CPU
    256.000000  255.972655  255.501468  255.623747   spread 1.95e-03
    truth 255.682232

Those are the `pp` row of the cross-rank table to three decimals (256.000 / 255.973 /
255.505 / 255.631, truth 255.682226), with no distribution at all. So the float64-combined
single-process form and the real all-reduced pipeline split agree, which makes the table a
measurement rather than an artifact of how the ranks were wired.

**And it is why the earlier 254.x numbers are retired.** They were measured on torch
2.8.0+cpu; `torch.randn` produces a different bfloat16 stream on 2.14 from the same seed,
so the same script gives 256.0000 / 255.9727 / 255.5015 / 255.6237 there. A maintainer
pasting a block would not have matched the table above it. Every number in the body is now
from 2.14.

Two details from that run that still apply:

* Stock torch raises `_get_total_norm() got an unexpected keyword argument 'dtype'` on the
  patched line, so a "needs this PR" marker in any snippet is accurate rather than
  decorative.
* Hand-rolling the same idea -- rounding the per-group norms yourself and combining them --
  shows NO spread, because CPU `vector_norm` upcasts internally. It reproduces only when
  each group's norm comes from `get_total_norm` itself.

## Blockers before this can be filed (2026-08-18)

1. **The guard uses a string type check.** `type(t).__name__ == "DTensor"` matches any class
   with that name and is not how core identifies a subclass. `torch/nn/utils/clip_grad.py`
   cannot import `torch.distributed.tensor` at module scope, so the options are a local
   import inside the branch, or testing membership in `_foreach_supported_types` minus
   `torch.Tensor`. A reviewer will raise this; decide it before filing rather than in review.
2. **The narrower fix may be the one they want.** The actual gap is that
   `torch._foreach_norm`'s dtype overload has no DTensor dispatch rule. Adding that rule
   fixes it for every caller instead of working around it in `clip_grad.py`. The guard
   unblocks this PR; the dispatch rule is the better change and is worth offering in the
   description.
3. **The fork branch `ba370ede20` is not this patch.** Its docstring claims the norm is
   "returned in the tensors' dtype regardless", which contradicts its own code -- measured,
   bf16 input with `dtype=float32` returns float32 -- and it does not carry the empty-input
   fix or the DTensor guard. It must be updated before a PR is opened from it.
4. **The repro link needs the pushed SHA**, and the body's `REPLACE_SHA` filled in.>>>>>>> f619302 (Raising_PRs/PR26: the DTensor reading was wrong, and the body follows the patch again)
