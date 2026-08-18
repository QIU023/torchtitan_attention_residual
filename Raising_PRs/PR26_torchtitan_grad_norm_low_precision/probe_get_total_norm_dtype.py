# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Probe and test for a ``dtype`` argument on ``torch.nn.utils.get_total_norm``.

    python probe_get_total_norm_dtype.py                      # stock torch
    python probe_get_total_norm_dtype.py --module <clip_grad.py>   # a patched copy

``--module`` loads one file and uses its ``_get_total_norm``, so a patched
``clip_grad.py`` can be exercised without touching an installed torch.

Three jobs:

1. answer the kit's open question -- does ``torch._foreach_norm`` accept an explicit
   ``dtype=None``? The patch passes ``dtype`` unconditionally, so if it does not, that
   branch needs a conditional kwarg.
2. show the defect: the same tensors, grouped differently, give different totals.
3. on a patched function, assert the grouping dependence is gone.

Why the groups are combined in float64 here
-------------------------------------------
The defect is that each GROUP's norm is rounded before the groups are combined. Under
PP or EP the groups are ranks, and the cross-rank combine is a separate step, so the
per-group rounding is what survives into the total. Combining the partials in bfloat16
instead hides it: at magnitude ~158 the bf16 grid spacing is 2.0, so every grouping
snaps to the same representable value and the table reads as if nothing is wrong. The
first version of this probe did exactly that and reported no spread.

Why the patched assertion is a tolerance and not equality
---------------------------------------------------------
fp32 grouping differences do not vanish, they shrink: each group's norm is still
rounded, just at 2^-23 instead of 2^-8. Asserting bitwise equality would fail on a
correct patch. The assertion is that the spread drops below 1e-6 relative, against a
bf16 spread three orders of magnitude larger, and both numbers are printed so the
claim is checkable rather than taken on trust.
"""

import argparse
import importlib.util
import torch


SEED = 0
N_TENSORS = 394
NUMEL = 64
CUTS = (100, 200, 300)
FP32_SPREAD_CEILING = 1e-6


def load_get_total_norm(module_path):
    """The stock function, or ``_get_total_norm`` from one file."""
    if module_path is None:
        return torch.nn.utils.get_total_norm, f"torch {torch.__version__} (stock)"
    spec = importlib.util.spec_from_file_location("_patched_clip_grad", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._get_total_norm, f"torch {torch.__version__} + {module_path}"


def tensors():
    torch.manual_seed(SEED)
    return [torch.randn(NUMEL, dtype=torch.bfloat16) for _ in range(N_TENSORS)]


def exact_norm(ts):
    """The value every grouping should agree on."""
    return torch.linalg.vector_norm(
        torch.cat([t.to(torch.float64).flatten() for t in ts]), 2
    )


def probe_foreach_norm_dtype():
    x = [torch.randn(8, dtype=torch.bfloat16) for _ in range(3)]
    out = {}
    for label, kwargs in (
        ("omitted", {}),
        ("dtype=None", {"dtype": None}),
        ("dtype=torch.float32", {"dtype": torch.float32}),
    ):
        try:
            out[label] = str(torch._foreach_norm(x, 2, **kwargs)[0].dtype)
        except Exception as err:  # noqa: BLE001 -- reporting, not handling
            out[label] = f"{type(err).__name__}: {err}"
    return out


def supports_dtype(get_total_norm) -> bool:
    try:
        get_total_norm([torch.randn(4, dtype=torch.bfloat16)], 2.0, dtype=torch.float32)
    except TypeError:
        return False
    return True


def partition_table(get_total_norm, ts, dtype=None):
    """(label, total) per grouping, partials combined in float64.

    float64 for the combine so the ONLY rounding under test is the per-group one.
    """
    kwargs = {} if dtype is None else {"dtype": dtype}
    rows = [("one group", get_total_norm(ts, 2.0, **kwargs).to(torch.float64))]
    for cut in CUTS:
        partials = torch.stack(
            [
                get_total_norm(ts[:cut], 2.0, **kwargs).to(torch.float64),
                get_total_norm(ts[cut:], 2.0, **kwargs).to(torch.float64),
            ]
        )
        rows.append((f"split {cut}/{N_TENSORS - cut}", torch.linalg.vector_norm(partials, 2)))
    return rows


def spread(rows):
    """Max relative difference between any two groupings of the same tensors."""
    values = [v.item() for _, v in rows]
    return (max(values) - min(values)) / max(values)


def report(rows, exact):
    for label, value in rows:
        err = abs(value.item() - exact.item()) / exact.item() * 100
        print(f"  {label:<20} {value.item():.6f}   {err:.3f}% from exact")
    print(f"  {'spread across groupings':<20} {spread(rows):.3e} relative")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default=None, help="path to a patched clip_grad.py")
    args = ap.parse_args()

    get_total_norm, where = load_get_total_norm(args.module)
    print(where)
    failures = []

    print("\n-- torch._foreach_norm dtype support --")
    for label, result in probe_foreach_norm_dtype().items():
        print(f"  {label:<22} -> {result}")

    ts = tensors()
    exact = exact_norm(ts)
    print(f"\n-- float64 reference --\n  {exact.item():.6f}")

    print("\n-- bfloat16 accumulation (today) --")
    bf16_rows = partition_table(get_total_norm, ts)
    report(bf16_rows, exact)
    bf16_spread = spread(bf16_rows)

    if not supports_dtype(get_total_norm):
        print(
            "\nThis get_total_norm has no dtype argument, so the patched arm is skipped."
            "\nThe spread above is the defect: same tensors, different grouping,"
            "\ndifferent total. Under PP or EP the grouping is where the model was cut."
        )
        return 0

    print("\n-- float32 accumulation (patched) --")
    fp32_rows = partition_table(get_total_norm, ts, dtype=torch.float32)
    report(fp32_rows, exact)
    fp32_spread = spread(fp32_rows)

    if fp32_spread > FP32_SPREAD_CEILING:
        failures.append(
            f"grouping dependence survives: fp32 spread {fp32_spread:.3e} "
            f"> {FP32_SPREAD_CEILING:.0e}"
        )
    if not fp32_spread < bf16_spread:
        failures.append(
            f"fp32 spread {fp32_spread:.3e} is not below bf16's {bf16_spread:.3e}"
        )

    got = get_total_norm(ts, 2.0, dtype=torch.float32).dtype
    if got is not torch.float32:
        failures.append(f"returned dtype is {got}, expected torch.float32")

    got = get_total_norm(ts, 2.0).dtype
    if got is not torch.bfloat16:
        failures.append(f"default changed: returned {got}, expected torch.bfloat16")

    empty = get_total_norm([], 2.0)
    if empty.dtype is not torch.float32 or empty.item() != 0.0:
        failures.append(f"empty default changed: {empty!r}")
    if get_total_norm([], 2.0, dtype=torch.float64).dtype is not torch.float64:
        failures.append("empty case ignores dtype")

    # foreach=False is a separate branch; only the foreach one ran above.
    a = get_total_norm(ts, 2.0, foreach=False, dtype=torch.float32)
    b = get_total_norm(ts, 2.0, dtype=torch.float32)
    if not torch.equal(a, b):
        failures.append(f"foreach=False disagrees with foreach: {a} != {b}")

    print(f"\n-- assertions ({bf16_spread / max(fp32_spread, 1e-18):.0f}x tighter) --")
    if failures:
        for f in failures:
            print(f"  FAIL {f}")
    else:
        print("  all passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
