# PR26 verification, 2026-08-18 (Linux GPU box)

Verifying the CPU-only draft from the Windows box. torch here is **2.14.0.dev+cu130**; the
Windows box had 2.8. Different major, same conclusions where checked -- a useful second
version point for a PR that targets upstream main.

The handoff commit `0dabe8f` and `GPU_HANDOFF.md` were NOT reachable from this box (origin
main is at `a779b52`, and the file is absent), so this ran off the task description plus the
committed `grad_norm_fp32_PR26.patch` and `PR.md`, which carry everything needed. The patch
is already applied on `dep_exp_impl` (`_get_total_norm_fp32` present), so probes import it
directly.

## Priority 1 (highest): DTensor placement -- the claim that was never run

The helper docstring says "the dtype argument preserves ``_NormPartial``". `PR.md`'s Fix
paragraph rests on it. `probe_dtensor_placement.py`, 2 ranks, gloo, no model:

```
[1] vector_norm of a Shard(0) DTensor, no dtype vs dtype=float32:
  no dtype   : DTensor placements=(_NormPartial(2.0),) dtype=bfloat16
  dtype=fp32 : DTensor placements=(_NormPartial(2.0),) dtype=float32
  -> dtype preserves the placement: True

[2] _get_total_norm_fp32 on [Shard(0), Replicate()] DTensors:
  total_norm : DTensor placements=(Replicate(),) dtype=float32
  total (full_tensor) = 16.514862   fp32 reference = 16.514862   rel err 0.000e+00
```

**CONFIRMED.** `dtype=float32` keeps the `_NormPartial(2.0)` placement, so the cross-rank
combine still uses the norm rule rather than a sum, and the end-to-end result matches the
fp32 reference. The PR body's claim holds, on torch 2.14.

## Priority 3: foreach=True explicit, and mixed-dtype inputs

`probe_foreach_mixed.py`, single process CPU. The risks were: passing `dtype` to
`torch._foreach_norm` might raise, and mixed bf16+fp32 input (a few params kept in fp32
under mixed precision) might choke the stack once `_group_tensors_by_device_and_dtype`
splits by dtype.

Neither happens. Every case runs, and the fp32 result is **invariant across
foreach None/True/False** (spread 0.00e+00) on the mixed input -- which is the real
assertion, not a tautology: if `dtype` failed to reach the foreach kernel, the `True` arm
would compute in bf16 and diverge from the `False` (per-tensor fp32) arm. It does not, so
`dtype` reaches both paths.

## The trap-1 dead zone, hit and mapped

The handoff warns a wrong probe passes by snapping bf16 groupings to one value. My first
negative control did exactly that: at norm ~69 the bf16 grid spacing is 0.5 and all four
groupings read 69.00000.

More than a caught mistake, this is a finding about where the bug is reproducible. A naive
CPU emulation -- stack bf16 per-tensor norms, `vector_norm` again -- **cannot** show the
grouping spread at any magnitude I swept (788/394/256 tensors, norms 1.8 to 15.9): all dead.
The reason is that CPU `vector_norm` upcasts internally, so the only bf16 rounding is the
final cast, which grouping does not move.

Forcing the accumulation to stay in bf16 (squares and sums kept bf16, no upcast -- what a
cross-rank reduce does) reproduces it sharply:

```
bf16-accumulated, grouped 1/2/4/8:  8.0000  11.3137  11.0227  11.2027   spread 3.3137
fp32 true:                          11.2254
```

So the grouping-dependence is real, and it is only visible where the reduction genuinely
stays in bf16 -- a distributed reduce, or forced accumulation -- NOT a single-process CPU
`vector_norm`. That is why the PR body's evidence is a 4-GPU run and the discovery was two
pipeline layouts, not a CPU snippet. **Worth adding to the PR notes**: anyone trying to
reproduce on CPU with plain tensors will see nothing and wrongly conclude there is no bug.

## Priority 4: pytorch's own test coverage -- located to the module, not the test

`get_total_norm` is defined in `torch/nn/utils/clip_grad.py` (confirmed via
`inspect.getsourcefile`). The test files are not shipped with the wheel, so the test that
covers it cannot be located from this box. It lives in the pytorch source repo, not here;
naming a specific test file would be a guess, which the task said not to make. To be filled
in from a pytorch checkout before citing coverage.

## Priority 2: CUDA -- DONE

Same probes on CUDA (`probe_*_cuda.py`), matrix aborted to free the GPU.

DTensor, 2 ranks nccl: `_NormPartial` preserved in float32, end-to-end total matches the
fp32 reference with relative error **1.06e-07** -- nonzero, unlike CPU's exact 0, which is
the signature of a real cross-rank nccl reduce rather than a local computation, and well
inside tolerance.

foreach/mixed on CUDA: identical shape to CPU -- no raise, foreach-invariant on mixed input.

### The demonstration a reviewer will want, through the real upstream function

512 bf16 CUDA grads, grouped k=1/2/4/8 the way PP/EP splits them, each group's norm taken by
**`torch.nn.utils.get_total_norm` itself** (unpatched upstream), then combined; against the
patched fp32 helper:

```
grouping           k=1      k=2      k=4      k=8    spread
upstream (bf16) : 12.8125  12.7500  12.8125  12.8125   0.0625
patched  (fp32) : 12.7840  12.7840  12.7840  12.7840   0.0000
fp32 truth      : 12.7840
```

Upstream's reported norm changes with the grouping -- k=2 reads 12.7500 against 12.8125 for
the others, one bf16 grid step (2^-4 = 0.0625 at this magnitude). The patch removes the
dependence and lands exactly on the fp32 truth. This is the PR's premise reproduced through
the actual `get_total_norm`, on CUDA, torch 2.14 -- not an emulation.

## State

Nothing pushed, nothing filed. The PASTE blocks in `PR.md` are unchanged and still need a
human nod. Two probes added to this folder. The one substantive addition the verification
suggests for the PR is the CPU-irreproducibility note above, so a reviewer who tries plain
CPU tensors is not misled.
