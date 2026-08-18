# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""`dtype=None` must reproduce the current result BITWISE, not approximately.

The PR body says "dtype=None is current behaviour". That is the claim a reviewer of an
API addition actually cares about, and until now it was asserted rather than evidenced --
the other probes all measure the arm where the behaviour is SUPPOSED to change.

Bitwise is the right judge here, and it is the opposite call from
`probe_get_total_norm_dtype.py`, deliberately: with `dtype=None` the patched function must
execute the same ops in the same order on the same values, so any difference at all is a
defect. With `dtype=float32` it is a different computation and only a tolerance makes
sense.

    python probe_default_unchanged.py --module <patched clip_grad.py>

Also verifies the exact snippet the PR body quotes, so the numbers in the body and the
numbers a reviewer gets from pasting it cannot drift apart.
"""

import argparse
import importlib.util

import torch


DTYPES = (torch.bfloat16, torch.float16, torch.float32, torch.float64)
FOREACH = (None, True, False)
NORM_TYPES = (1.0, 2.0, float("inf"))
SHAPES = (
    ("empty", 0, 0),
    ("single", 1, 64),
    ("many", 37, 64),
    ("ragged", 8, 129),
)


def load_patched(path):
    spec = importlib.util.spec_from_file_location("_patched_clip_grad", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._get_total_norm


def make(n, numel, dtype, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(numel, dtype=dtype) for _ in range(n)]


def check_default_unchanged(patched):
    stock = torch.nn.utils.get_total_norm
    cases = failures = 0
    for label, n, numel in SHAPES:
        for dtype in DTYPES:
            for foreach in FOREACH:
                for norm_type in NORM_TYPES:
                    ts = make(n, numel, dtype)
                    try:
                        want = stock(ts, norm_type, False, foreach)
                    except Exception as err:  # noqa: BLE001
                        want = f"{type(err).__name__}"
                    try:
                        got = patched(ts, norm_type, False, foreach, dtype=None)
                    except Exception as err:  # noqa: BLE001
                        got = f"{type(err).__name__}"
                    cases += 1
                    if isinstance(want, str) or isinstance(got, str):
                        same = want == got
                    else:
                        same = (
                            want.dtype == got.dtype
                            and want.shape == got.shape
                            and torch.equal(want, got)
                        )
                    if not same:
                        failures += 1
                        print(
                            f"  DIFFER {label:<7} {str(dtype):<16} "
                            f"foreach={str(foreach):<5} p={norm_type}: "
                            f"{want!r} vs {got!r}"
                        )
    return cases, failures


def body_snippet():
    """Exactly what the PR body tells a reviewer to paste. Returns its printed rows."""
    gtn = torch.nn.utils.get_total_norm
    torch.manual_seed(0)
    grads = [torch.randn(128, dtype=torch.bfloat16) for _ in range(512)]
    truth = torch.linalg.vector_norm(torch.cat([g.double() for g in grads]), 2).item()

    def total(k, **kw):  # k round-robin groups, partials combined in float64
        parts = torch.stack([gtn(grads[i::k], 2.0, **kw).double() for i in range(k)])
        return torch.linalg.vector_norm(parts, 2).item()

    return [total(k) for k in (1, 2, 4, 8)], truth


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="path to a patched clip_grad.py")
    args = ap.parse_args()
    patched = load_patched(args.module)
    print(f"torch {torch.__version__}\n")

    print("-- dtype=None must be bitwise identical to today --")
    cases, failures = check_default_unchanged(patched)
    print(f"  {cases - failures}/{cases} cases bitwise identical "
          f"({len(SHAPES)} shapes x {len(DTYPES)} dtypes x {len(FOREACH)} foreach "
          f"x {len(NORM_TYPES)} norm types)")

    print("\n-- the snippet quoted in the PR body, run here --")
    rows, truth = body_snippet()
    print("  today          " + "  ".join(f"{v:9.4f}" for v in rows))
    print(f"  float32 truth  {truth:9.4f}")
    patched_rows = []
    gtn_p = patched
    torch.manual_seed(0)
    grads = [torch.randn(128, dtype=torch.bfloat16) for _ in range(512)]
    for k in (1, 2, 4, 8):
        parts = torch.stack(
            [gtn_p(grads[i::k], 2.0, dtype=torch.float32).double() for i in range(k)]
        )
        patched_rows.append(torch.linalg.vector_norm(parts, 2).item())
    print("  dtype=float32  " + "  ".join(f"{v:9.4f}" for v in patched_rows))

    print("\n-- assertions --")
    if failures:
        print(f"  FAIL {failures} case(s) changed at dtype=None")
    else:
        print("  dtype=None unchanged in every case")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
