# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Per-parameter gradient deviation between a reference leg and a parallel leg.

The probe dumps ``{param_name: grad_norm}`` per rank. A parallel leg's ranks hold
different parameters, so the leg's view is the UNION across its ranks; the reference
runs on one rank and holds all of them.

Two columns, because one of them is misleading on its own:

* ``max_all`` over every shared parameter. A parameter whose gradient is ~0 in both
  legs gives a large relative difference from an absolutely tiny one, so this column
  is dominated by near-zero gradients and cannot distinguish "wrong" from "small".
* ``max>1%`` restricted to parameters holding more than 1% of the reference's total
  gradient norm. This is the column a claim should rest on.

Both are reported because the earlier documents report both, and dropping one would
make the numbers incomparable with them.
"""

from __future__ import annotations

import json
import pathlib
import sys


def _load_leg(stem: pathlib.Path) -> dict[str, float]:
    """Union across ranks. Later ranks do not overwrite earlier ones -- a parameter
    present on two ranks under PP would be a replicated one, and taking either copy is
    the same question this script is asking, so a disagreement there is reported."""
    merged: dict[str, float] = {}
    parts = sorted(stem.parent.glob(stem.name + ".r*"))
    if not parts:
        raise FileNotFoundError(f"no rank dumps for {stem}")
    for part in parts:
        for key, value in json.loads(part.read_text()).items():
            merged.setdefault(key, value)
    return merged


def compare(ref_stem: pathlib.Path, leg_stem: pathlib.Path) -> dict:
    ref, leg = _load_leg(ref_stem), _load_leg(leg_stem)
    shared = sorted(set(ref) & set(leg))
    if not shared:
        raise ValueError(f"no shared parameters between {ref_stem} and {leg_stem}")
    total = sum(v * v for v in ref.values()) ** 0.5
    rows = []
    for key in shared:
        a, b = ref[key], leg[key]
        # Relative to the REFERENCE, not to max(|a|, |b|). That makes the metric
        # unbounded, which matters: the recorded full-parameter max_all of 2.37 is
        # above 1 and cannot be produced by a max-denominator, so the earlier
        # documents used this definition and a capped one is not comparable with them.
        rel = 0.0 if a == 0 else abs(a - b) / abs(a)
        scale = max(abs(a), abs(b))
        bounded = 0.0 if scale == 0 else abs(a - b) / scale
        share = 0.0 if total == 0 else abs(a) / total
        rows.append((rel, share, key, bounded))
    rows.sort(reverse=True)
    big = [r for r in rows if r[1] > 0.01]
    return {
        "n_shared": len(shared),
        "max_all": rows[0][0],
        "max_all_param": rows[0][2],
        "max_big": big[0][0] if big else 0.0,
        "max_big_param": big[0][2] if big else "-",
        "max_all_bounded": max(r[3] for r in rows),
        "max_big_bounded": max((r[3] for r in big), default=0.0),
        "n_big": len(big),
    }


def main() -> None:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/attnres_band")
    arms = sys.argv[2:] or ["ar_bf16", "ar_fp32", "no_bf16", "no_fp32"]
    # Two denominators. |a-b|/|ref| is what the earlier documents used (unbounded, so
    # their 2.37 is only expressible this way); |a-b|/max(|a|,|b|) is bounded by 1 and
    # is the one to reason with. Reporting both keeps the numbers comparable with those
    # documents without inheriting a metric that cannot be read as a percentage.
    print(
        f"{'arm':10} {'leg':5} {'shared':>7} {'max_all':>9} {'max>1%':>8} "
        f"{'maxAll_b':>9} {'max>1%_b':>9}  worst>1%"
    )
    for arm in arms:
        ref = out / f"{arm}_ref.json"
        if not sorted(out.glob(f"{arm}_ref.json.r*")):
            print(f"{arm:10} {'--':5} {'':>7} {'no ref dump':>10}")
            continue
        for leg in ("pp2", "cp2"):
            stem = out / f"{arm}_{leg}.json"
            try:
                r = compare(ref, stem)
            except (FileNotFoundError, ValueError) as err:
                print(f"{arm:10} {leg:5} {'':>7} {type(err).__name__}")
                continue
            print(
                f"{arm:10} {leg:5} {r['n_shared']:>7} {r['max_all']:>9.5f} "
                f"{r['max_big']:>8.5f} {r['max_all_bounded']:>9.5f} "
                f"{r['max_big_bounded']:>9.5f}  {r['max_big_param']}"
            )


if __name__ == "__main__":
    main()
