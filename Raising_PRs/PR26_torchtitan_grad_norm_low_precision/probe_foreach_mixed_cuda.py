"""foreach=True explicitly, and mixed-dtype inputs -- the two plain-tensor paths nobody ran.

The CPU numbers in the PR body all came from the default ``foreach=None`` with uniform bf16.
Two things the helper does that were never exercised:

* ``foreach=True`` forces ``torch._foreach_norm(tensors, norm_type, dtype=torch.float32)``.
  Passing ``dtype`` to the foreach kernel is the part that had no test.
* Mixed input dtypes. ``_group_tensors_by_device_and_dtype`` (which the foreach path uses
  internally) groups by dtype, so with some bf16 and some fp32 grads -- the ordinary case
  when a few params are kept in fp32 under mixed precision -- the norms that come back are
  heterogeneous WITHOUT ``dtype`` and homogeneous WITH it. The subsequent ``torch.stack``
  is what changes shape of problem, and nobody checked it does not choke.

No ranks needed; this is single-process CPU.

    python probe_foreach_mixed.py
"""

import torch

from torchtitan.distributed.utils import _get_total_norm_fp32


def reference(tensors):
    """The value the reduction should approach: every norm taken in fp32, then normed."""
    per = [torch.linalg.vector_norm(t.float(), 2.0) for t in tensors]
    return torch.linalg.vector_norm(torch.stack(per), 2.0).item()


torch.manual_seed(0)
TOL = 1e-2  # tolerance, not equality: fp32-over-bf16 narrows the gap, it does not erase it.

print("case                         foreach  result      fp32 ref    rel err   ok")
cases = {
    "uniform bf16": [torch.randn(64, dtype=torch.bfloat16, device="cuda") for _ in range(8)],
    "uniform fp32": [torch.randn(64, device="cuda") for _ in range(8)],
    # A few params kept in fp32 while the rest train in bf16 -- ordinary mixed precision.
    "mixed bf16+fp32": (
        [torch.randn(64, dtype=torch.bfloat16, device="cuda") for _ in range(6)]
        + [torch.randn(64, device="cuda") for _ in range(2)]
    ),
}

all_ok = True
for name, grads in cases.items():
    ref = reference(grads)
    for foreach in (None, True, False):
        try:
            out = _get_total_norm_fp32(
                grads, 2.0, error_if_nonfinite=False, foreach=foreach
            ).item()
            rel = abs(out - ref) / max(ref, 1e-12)
            ok = rel < TOL
        except Exception as e:  # a raised kernel error is the failure this probe hunts
            out, rel, ok = float("nan"), float("nan"), False
            print(f"  {name:24s} foreach={foreach}: RAISED {type(e).__name__}: {e}")
            all_ok = False
            continue
        all_ok &= ok
        fe = {None: "None ", True: "True ", False: "False"}[foreach]
        print(f"  {name:24s} {fe}   {out:9.5f}   {ref:9.5f}   {rel:.2e}  {'ok' if ok else 'BAD'}")

# The point of the mixed case is comparative, not absolute: with dtype=fp32 forced, the
# result must NOT depend on foreach, because the whole reduction is fp32 either way. A
# difference between foreach arms on the same input would mean the dtype is not actually
# reaching one of the kernels.
print("\nforeach-invariance on the mixed case (fp32 reduction must not depend on foreach):")
grads = cases["mixed bf16+fp32"]
vals = {
    fe: _get_total_norm_fp32(grads, 2.0, error_if_nonfinite=False, foreach=fe).item()
    for fe in (None, True, False)
}
spread = max(vals.values()) - min(vals.values())
print(f"  None={vals[None]:.6f}  True={vals[True]:.6f}  False={vals[False]:.6f}  spread={spread:.2e}")
inv_ok = spread < 1e-4
all_ok &= inv_ok
print(f"  foreach-invariant: {inv_ok}")

print(f"\nPROBE {'PASS' if all_ok else 'FAIL'}")
