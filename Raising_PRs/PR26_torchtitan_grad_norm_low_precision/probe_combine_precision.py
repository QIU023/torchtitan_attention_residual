# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""What makes the grouping-dependence visible: the COMBINE dtype, not the device.

`VERIFY_RESULTS_2026-08-18.md` concluded that a single-process CPU emulation "cannot
show the grouping spread at any magnitude I swept", and proposed telling reviewers so in
the PR. This script refutes that, so it is not told.

The per-group norm is bf16 in every row below; only the dtype the partials are combined
in changes. Combined at bf16 the effect is quantized to the bf16 grid and CAN vanish for
a particular fixture -- that is the dead zone, and it is fixture-dependent, not
magnitude-monotone. Combined at any higher precision it is present in every fixture at
~2e-3 relative, on plain CPU tensors, single process.

The second half of that document's explanation is also the wrong way round. CPU
`vector_norm` does upcast internally, so each group's norm is accumulated in fp32 and
rounded to bf16 once at the end -- and THAT final per-group rounding is exactly what
grouping moves, because different groupings round different partial sums. It is the
mechanism, not a reason the mechanism is invisible.

    python probe_combine_precision.py
"""

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


def tensors(n, numel, scale, seed=0):
    torch.manual_seed(seed)
    return [torch.randn(numel, dtype=torch.bfloat16) * scale for _ in range(n)]


def exact(ts):
    return torch.linalg.vector_norm(
        torch.cat([t.to(torch.float64).flatten() for t in ts]), 2
    ).item()


def grouped_total(ts, k, combine):
    """Each group's norm in its own dtype (bf16), combined in ``combine``.

    Round-robin groups rather than contiguous ones: a pipeline stage owns every k-th
    layer under an interleaved schedule, and contiguous vs round-robin is itself a
    grouping difference, which is the thing under test.
    """
    get_total_norm = torch.nn.utils.get_total_norm
    partials = torch.stack(
        [get_total_norm(ts[i::k], 2.0).to(combine) for i in range(k)]
    )
    return torch.linalg.vector_norm(partials, 2).item()


def main() -> int:
    print(f"torch {torch.__version__}")
    print("per-group norm is bfloat16 in every row; only the COMBINE dtype differs")
    print(f"groupings k = {', '.join(str(k) for k in GROUPINGS)} (round-robin)\n")

    any_alive_at_bf16 = False
    all_alive_above_bf16 = True

    for n, numel, scale in FIXTURES:
        ts = tensors(n, numel, scale)
        ref = exact(ts)
        print(f"n={n:<4} numel={numel:<4} exact={ref:10.4f}")
        for combine in (torch.bfloat16, torch.float32, torch.float64):
            vals = [grouped_total(ts, k, combine) for k in GROUPINGS]
            sp = (max(vals) - min(vals)) / max(vals)
            name = str(combine).replace("torch.", "")
            note = ""
            if sp == 0.0:
                note = "   <-- dead, every grouping equal"
                if combine is not torch.bfloat16:
                    all_alive_above_bf16 = False
            elif combine is torch.bfloat16:
                any_alive_at_bf16 = True
            print(
                f"   combine {name:<9} spread {sp:.3e}   "
                + " ".join(f"{v:.4f}" for v in vals)
                + note
            )
        print()

    print("conclusions")
    print(f"  above bf16, every fixture shows the spread : {all_alive_above_bf16}")
    print(f"  at bf16, some fixtures show it and some do not : {any_alive_at_bf16}")
    print(
        "\nSo the defect IS reproducible single-process on CPU with plain tensors.\n"
        "What has to be said when quoting a number is how the partials were combined."
    )
    return 0 if all_alive_above_bf16 else 1


if __name__ == "__main__":
    raise SystemExit(main())
