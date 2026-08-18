# repro table + a defect the repro found that the earlier probes missed

`repro_get_total_norm_dtype.py`, torch 2.14 CUDA/nccl, 512 bf16 grads, worlds 1/2/4/8.

## The defect, first: the patch crashes on DTensor + dtype

`repro` exercises the REAL `torch.nn.utils.get_total_norm`, and on the `dtensor fp32` arm
the patch as written -- BOTH `ba370ede20` (fork branch) and the repo's Windows version --
raises:

    RuntimeError: '>=' not supported between instances of 'torch.dtype' and 'int'

Root cause, isolated: DTensors pass `_has_foreach_support`, so upstream `_get_total_norm`
routes them to `torch._foreach_norm(tensors, norm_type, dtype=dtype)`. That call works
WITHOUT dtype (returns `_NormPartial`) but DTensor's dispatch has no `_foreach_norm.dtype`
overload -- it mis-parses `dtype` as a dim. `torch.linalg.vector_norm(dtensor, ..., dtype=)`
handles it fine.

The earlier probes missed this because `probe_dtensor_placement.py` tested the TORCHTITAN
helper `_get_total_norm_fp32`, which forces DTensors down the per-tensor `vector_norm` path
(`any(isinstance(t, DTensor))`). The upstream patch has no such guard, so it breaks on
exactly the input the PR targets -- FSDP gradients, which are DTensors. **The repro caught
what a probe of the wrong function could not.**

### Fix the patch needs before it can be filed

Route DTensors to the per-tensor `vector_norm` path when `dtype` is set -- the guard the
torchtitan helper already carries and this repo already validated. In `_get_total_norm`'s
per-group loop:

    use_foreach = (<existing foreach condition>) and not (dtype is not None and <group has DTensor>)

with the `elif foreach:` refined so the DTensor+dtype fallback does not trip its
"can't use foreach" raise. (The narrower alternative is a DTensor `_foreach_norm.dtype`
dispatch rule in pytorch core; the guard is the one that unblocks the PR now.)

The table below was produced WITH that guard applied to the repo patch. The raw repo patch
does not produce it -- it crashes on the second row.

## Table (guarded patch)

```
                world=1      world=2      world=4      world=8      spread(rel)
dtensor today   256.000000   256.000000   256.000000   256.000000   0.00e+00
dtensor fp32    255.682220   255.682220   255.682220   255.682220   0.00e+00
pp today        256.000000   255.972656   255.505386   255.630600   1.93e-03
pp fp32         255.682220   255.682220   255.682236   255.682220   5.97e-08
float64 truth   255.682226
```

Reading:

* **`pp today`** is world-dependent -- 256.000 at world 1, 255.505 at world 4, relative
  spread 1.93e-3 (0.19%). Same gradients, different pipeline split, different reported norm.
  This is the bug the PR fixes.
* **`pp fp32`** collapses the spread to 5.97e-8 and matches the float64 truth 255.682226.
* **`dtensor today`** is world-INDEPENDENT (the `_NormPartial` reduce is consistent) but
  bf16-imprecise: 256.000 against the truth 255.682. So DTensor did not have the grouping
  bug, but the patch still improves its accuracy.
* **`dtensor fp32`** lands on the truth.

So the two splits fail differently: PP mismeasures AND varies with the split; DTensor
mismeasures consistently. Both are corrected by the dtype accumulation, once the DTensor
path actually reaches `vector_norm` instead of crashing in `_foreach_norm`.

## Also confirmed: the return-dtype contradiction the task flagged

`ba370ede20`'s docstring says "The norm is returned in the tensors' dtype regardless."
Measured: bf16 input with `dtype=float32` returns **float32**, not bf16 -- because the patch
threads `dtype` into the final norm-of-norms too. The repo/Windows version's docstring and
its `return torch.tensor(0.0, dtype=dtype)` for the empty case are the correct ones. The
fork branch `ba370ede20` should be updated to the repo version (or the docstring corrected)
before filing.
