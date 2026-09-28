"""Per-rank peak vs width from three 5060 runs of the widened debug model, extrapolated to H100 widths.

Usage: python fit3.py <res_dir> <prefix> <step> <key> <dim> [<dim> ...]
Each rank's peak is fitted exactly as A + B s + C s^2 over s = dim / 256 at dims 512, 768, 1024
(parameters and optimizer states grow as s^2, activations as s), then evaluated at the given dims.
"""
import json
import os
import sys

res, prefix, step, key = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
targets = [int(x) for x in sys.argv[5:]]
points = (512, 768, 1024)


def peaks(dim):
    folder = os.path.join(res, f"{prefix}_d{dim}", "mem")
    out = {}
    for name in sorted(os.listdir(folder)):
        data = json.load(open(os.path.join(folder, name)))
        out[data["rank"]] = data["records"][step][key]
    return out


meas = {d: peaks(d) for d in points}
ranks = sorted(meas[points[0]])
print(f"{prefix} {key} at step {step}, measured on 5060 (GiB):")
for d in points:
    print(f"  dim {d}: " + " ".join(f"{meas[d][r]:.2f}" for r in ranks))
fits = {}
for r in ranks:
    (s0, s1, s2) = (d / 256 for d in points)
    y0, y1, y2 = (meas[d][r] for d in points)
    # Lagrange form of the quadratic through the three points
    fits[r] = lambda s, s0=s0, s1=s1, s2=s2, y0=y0, y1=y1, y2=y2: (
        y0 * (s - s1) * (s - s2) / ((s0 - s1) * (s0 - s2))
        + y1 * (s - s0) * (s - s2) / ((s1 - s0) * (s1 - s2))
        + y2 * (s - s0) * (s - s1) / ((s2 - s0) * (s2 - s1))
    )
for d in targets:
    p = [fits[r](d / 256) for r in ranks]
    print(f"  predicted dim {d}: " + " ".join(f"{x:.1f}" for x in p) + f" | max {max(p):.1f}")
