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

This passes a `dtype` through to the three norm calls the function already makes; `torch.linalg.vector_norm` and `torch._foreach_norm` both take `dtype` already.

One fix underneath is needed first, as a separate commit here. DTensor gradients -- the FSDP case -- reach `torch._foreach_norm(..., dtype=)`, and that raises today: `_foreach_norm.Scalar` shares `vector_norm`'s DTensor sharding strategy, which reads `args_schema[2]` as `dim`, while `_foreach_norm`'s schema has no dim and position 2 is `dtype`. Setting `dim=None` for that overload fixes every `_foreach_norm(dtype=)` DTensor caller rather than only this one, and keeps `get_total_norm` a pure passthrough with no DTensor special-casing.

### The two splits

512 bf16 gradients, byte-identical at every world size -- the only thing that changes is
how they are divided across ranks. Both divisions come from real training:

* **FSDP.** Every gradient is a DTensor `Shard(0)` over the mesh. `get_total_norm` returns
  a DTensor carrying `_NormPartial`, and `full_tensor()` performs the cross-rank combine.
* **Pipeline.** Rank `r` owns `grads[r::world]` as plain tensors -- what a stage holds.
  Each rank norms its own, then the partials are all-reduced.

The reference is float64 over the whole set, computed identically on every rank, so it does
not depend on the world size. Neither should the reported norm.

### Reproducing

nccl, 8 GPUs, torch 2.14, both commits built:

```
python repro_get_total_norm_dtype.py --module torch/nn/utils/clip_grad.py --world 1,2,4,8
```

Script: https://github.com/QIU023/torchtitan_attention_residual/blob/315b1a247bd9f1f81526ae84f6a92373f096643e/Raising_PRs/PR26_torchtitan_grad_norm_low_precision/repro_get_total_norm_dtype.py

The two `today` rows need no patch and reproduce on stock torch -- drop `--module`. The
fp32 DTensor row needs the DTensor commit built in, not just the `clip_grad.py` change;
without it that case raises instead of printing, which is what the commit is for:

```
RuntimeError: '>=' not supported between instances of 'torch.dtype' and 'int'
  torch._foreach_norm(device_tensors, norm_type, dtype=dtype)
  -> _propagate_op_sharding_dispatch_slow_path
```

### Result

```
                world=1    world=2    world=4    world=8    spread(rel)
dtensor today   256.000    256.000    256.000    256.000    0.00e+00
dtensor fp32    255.682    255.682    255.682    255.682    0.00e+00
pp today        256.000    255.973    255.505    255.631    1.93e-03
pp fp32         255.682    255.682    255.682    255.682    5.97e-08
float64 truth   255.682226
```

The pipeline row is the defect: the same gradients report a different norm depending only
on how many stages the model was cut into, so clipping takes a different-sized step for a
reason that has nothing to do with the gradients. The DTensor row does not vary with the
world size but sits 0.12% off the truth, bf16 having snapped the total to 256. Passing
`dtype=torch.float32` puts both on the truth.

### Default unchanged

`dtype=None` is bitwise identical to today in all 144 cases -- 4 shapes including empty, x
{bf16, fp16, fp32, fp64}, x foreach {None, True, False}, x p in {1, 2, inf}:

```
python probe_default_unchanged.py --module torch/nn/utils/clip_grad.py
```

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

Items 1 and 2 are CLOSED by taking route B -- the guard is gone and the dispatch fix is in
`dtensor_foreach_norm_dtype_pytorch.patch`. Verified against pytorch/main here:
`vector_norm_single_dim_strategy` takes `op` as its first parameter, so the new branch can
test it, and `aten._foreach_norm.Scalar` really is registered to that same strategy by a
separate `register_single_dim_strategy(...)` call below the function. The diagnosis holds.

Still open:

1. ~~A lint risk in the reverted passthrough~~ -- **WRONG, retracted.** I assumed 88 was the
   gate. `E501` is explicitly ignored in BOTH `pyproject.toml` and `.flake8` ("E501 is not
   flexible enough, we're using B950 instead"), and flake8's `max-line-length = 120`, so
   B950 permits 132. The longest line these patches introduce is 94. No lint problem.
2. ~~The fork branch is still not this patch~~ -- **resolved, and the reading was of a
   superseded line.** `ba370ede20` is not an ancestor of the branch head; it is the
   pre-split single commit with the same title. `git ls-remote` gives
   `refs/heads/get-total-norm-dtype = b198c32e5e`, which HAS the empty-input fix
   (`torch.tensor(0.0, dtype=dtype)` against `torch.tensor(0.0)`), HAS the dispatch commit
   `c54b24b7a7`, and whose docstring matches its code -- it says the per-tensor norms and
   the norm-of-norms accumulate in the given dtype, and all three call sites plus the empty
   case pass it. Do NOT push `ba370ede20`: it would revert both fixes.
3. ~~The repro link needs the pushed SHA~~ -- done, pinned to the full
   `315b1a247bd9f1f81526ae84f6a92373f096643e`, which is on `origin/main` and contains the
   script. Full sha rather than the short one, so it stays a permalink.
4. ~~Where the tests go~~ -- located. `test/test_nn.py::test_clip_grad_norm` for the dtype
   argument, and for the dispatch fix
   `test/distributed/tensor/test_math_ops.py::DistMathOpsTest`, which already has
   `test_foreach_norm` and `test_foreach_norm_partial` calling
   `torch.ops.aten._foreach_norm([...], 2)` on the same overload. So it is a `dtype=` case
   next to those, not a new file -- and their not having one is why the bug survived:
   without `dtype` there is nothing at `args_schema[2]` for the borrowed strategy to
   misread.
5. **The patch's `index` line is true of our base, not of today's main.** Regenerating it
   from the commit (`git diff c54b24b7a7^ c54b24b7a7 -- _math_ops.py`) reproduces the file
   byte for byte, so `d6f7a19147` is exactly the blob at the branch's parent. That parent is
   the fork's `main`, `6a34faa284` from 2026-08-06, while upstream main is 12 days ahead at
   `0e12e565cf` -- which is where `89ddcfeb2a` comes from. Nothing to correct in the header;
   what is wanted before filing is a rebase onto current upstream main, and this clone has
   no `pytorch/pytorch` remote to do it from.

## Final check against pytorch/main `0e12e565cf`, 2026-08-18

| check | result |
|---|---|
| both patches apply | CLEAN (see the CRLF note below) |
| `py_compile` on both touched files | ok |
| size | +20/-6 across 2 files |
| `vector_norm_single_dim_strategy` takes `op` first | YES, on current main |
| `aten._foreach_norm.Scalar` registered to that strategy | YES, by a separate `register_single_dim_strategy(...)` call below the function |
| `args_schema[2]` still read as `dim` | YES -- so the diagnosis holds on today's main |
| longest introduced line | 94 chars, under B950's 132 |

**The CRLF trap, found by this check.** On a Windows checkout of this logbook, `git apply`
rejects both patches with `patch does not apply` -- and `clip_grad.py` is byte-identical to
the blob the patch names, so the message points at the wrong thing entirely. The cause is
that git checks the `.patch` files out as CRLF here; the repo blobs are LF, so a Linux
checkout is unaffected. Fixed with a `.gitattributes` marking `*.patch` and `*.diff` as
`-text`. Anyone who hit this before would have concluded the patch was stale.
