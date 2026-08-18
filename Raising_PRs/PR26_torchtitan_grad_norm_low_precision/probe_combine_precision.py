# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The PR's evidence table, and what makes the effect visible at all.

    python probe_combine_precision.py
    python probe_combine_precision.py --device cuda --module <patched clip_grad.py>

Section 1 is the PR's headline table: bf16 gradients grouped 1/2/4/8 ways, each group's
norm taken by ``torch.nn.utils.get_total_norm`` itself, then combined. Today's total
moves with the grouping; with ``dtype=torch.float32`` it does not. ``--module`` supplies
the patched function for the second row; without it only the unpatched row is printed.

Section 2 is why a naive reproduction can miss it. ``VERIFY_RESULTS_2026-08-18.md``
concluded that a single-process CPU emulation "cannot show the grouping spread at any
magnitude I swept" and proposed telling reviewers so. It can, at ~2e-3 in every fixture,
as long as the partials are combined at higher precision than bf16. Combined AT bf16 the
effect is quantized to the grid and can vanish for a particular fixture -- and the dead
zone follows the fixture, not the magnitude.

That document's explanation is also inverted: CPU ``vector_norm`` upcasts internally, so
each group's norm is accumulated in fp32 and rounded to bf16 once at the end -- and that
per-group rounding is exactly what grouping moves, because different groupings round
different partial sums. It is the mechanism, not a reason the mechanism is invisible.
"""

import argparse
import importlib.util

import torch


GROUPINGS = (1, 2, 4, 8)
FIXTURES = (
    # (n_tensors, numel, scale) -- scale only moves the magnitude
    (394, 64, 1.0),
    (394, 64, 0.1),
    (788, 64, 1.0),
    (256, 64, 1.0),
    (512, 128, 1.0),
)


def load_patched(module_path):
    if module_path is None:
        return None
    spec = importlib.util.spec_from_file_location("_patched_clip_grad", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._get_total_norm


def tensors(n, numel, scale, device, seed=0):
    torch.manual_seed(seed)
    return [
        (torch.randn(numel, dtype=torch.bfloat16, device=device) * scale)
        for _ in range(n)
    ]


def exact(ts):
    return torch.linalg.vector_norm(
        torch.cat([t.to(torch.float64).flatten() for t in ts]), 2
    ).item()


def grouped_total(get_total_norm, ts, k, combine, dtype=None):
    """Each group's norm from ``get_total_norm``, then combined in ``combine``.

    Round-robin groups, not contiguous: under an interleaved schedule a pipeline stage
    owns every k-th layer. Contiguous vs round-robin is itself a grouping difference,
    which is the thing under test.
    """
    kwargs = {} if dtype is None else {"dtype": dtype}
    partials = torch.stack(
        [get_total_norm(ts[i::k], 2.0, **kwargs).to(combine) for i in range(k)]
    )
    return torch.linalg.vector_norm(partials, 2).item()


def spread(values):
    return (max(values) - min(values)) / max(values)


def headline(patched, device):
    """Section 1 -- the table the PR body quotes."""
    ts = tensors(512, 128, 1.0, device)
    ref = exact(ts)
    stock = torch.nn.utils.get_total_norm

    print(f"-- 512 bf16 tensors on {device}, grouped {GROUPINGS}, "
          f"combined in float64 --")
    rows = []
    vals = [grouped_total(stock, ts, k, torch.float64) for k in GROUPINGS]
    rows.append(("today (bf16)", vals))
    if patched is not None:
        rows.append(
            ("dtype=float32", [
                grouped_total(patched, ts, k, torch.float64, dtype=torch.float32)
                for k in GROUPINGS
            ])
        )
    width = max(len(name) for name, _ in rows)
    for name, vals in rows:
        cells = "  ".join(f"{v:9.4f}" for v in vals)
        print(f"  {name:<{width}}  {cells}   spread {spread(vals):.3e}")
    print(f"  {'fp32 truth':<{width}}  {ref:9.4f}")
    return rows


def sweep(device):
    """Section 2 -- the combine dtype decides visibility, not the device."""
    stock = torch.nn.utils.get_total_norm
    all_alive_above_bf16 = True
    any_dead_at_bf16 = False

    for n, numel, scale in FIXTURES:
        ts = tensors(n, numel, scale, device)
        print(f"  n={n:<4} numel={numel:<4} exact={exact(ts):10.4f}")
        for combine in (torch.bfloat16, torch.float32, torch.float64):
            vals = [grouped_total(stock, ts, k, combine) for k in GROUPINGS]
            sp = spread(vals)
            name = str(combine).replace("torch.", "")
            note = ""
            if sp == 0.0:
                note = "   <-- dead, every grouping equal"
                if combine is torch.bfloat16:
                    any_dead_at_bf16 = True
                else:
                    all_alive_above_bf16 = False
            print(f"    combine {name:<9} spread {sp:.3e}   "
                  + " ".join(f"{v:.4f}" for v in vals) + note)
    return all_alive_above_bf16, any_dead_at_bf16


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--module", default=None, help="path to a patched clip_grad.py")
    args = ap.parse_args()

    patched = load_patched(args.module)
    print(f"torch {torch.__version__}, device {args.device}, "
          f"patched function: {'yes' if patched else 'no'}\n")

    headline(patched, args.device)

    print("\n-- the combine dtype decides visibility; per-group norm is bf16 "
          "in every row --")
    all_alive, any_dead = sweep(args.device)

    print("\nconclusions")
    print(f"  above bf16, every fixture shows the spread   : {all_alive}")
    print(f"  at bf16, at least one fixture shows nothing  : {any_dead}")
    print(
        "\nThe defect IS reproducible single-process on CPU with plain tensors.\n"
        "Whenever a number is quoted, say how the partials were combined."
    )
    return 0 if all_alive else 1


if __name__ == "__main__":
    raise SystemExit(main())
