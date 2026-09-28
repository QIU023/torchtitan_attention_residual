"""Pick the H100 width for a pp4 x vp8 probe layout from two 5060 runs at smaller widths.

Usage: PYTHONPATH=<kit>:<tree>:<attn_gym> python fit_width.py <res_dir> <step> <cell_lo> <dim_lo> <cell_hi> <dim_hi> <target_gib> [<dim> ...]
Per rank: peak(D) = 16 B x params on the rank at D (counted on meta, titan's split, stage s on rank s % 4)
+ a + b x D, with a and b fitted to the two measured peaks. Prints the largest multiple of 512 whose
predicted heaviest rank stays at or under the target, with every rank's prediction.
"""
import json
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch

res, step = sys.argv[1], int(sys.argv[2])
cell_lo, dim_lo, cell_hi, dim_hi = sys.argv[3], int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
target = float(sys.argv[7])
extra_dims = [int(x) for x in sys.argv[8:]]
PP, GiB = 4, 2**30
_static = {}


def static(dim):
    if dim in _static:
        return _static[dim]
    os.environ["PPMEM_DIM"] = str(dim)
    import importlib
    import probe_lb
    importlib.reload(probe_lb)
    from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part, _get_pipeline_metadata
    config = probe_lb.lb_probe()
    with torch.device("meta"):
        model = config.model.build()
    per_module = defaultdict(int)
    for name, p in model.named_parameters():
        parts = name.split(".")
        per_module[".".join(parts[:2]) if parts[0] == "layers" else parts[0]] += p.numel()
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    n, layers, iw, ow = _get_pipeline_metadata(SimpleNamespace(pp=PP), config.parallelism, config.model)
    split = _generate_llm_fqn_per_model_part(n, layers, iw, ow)
    placed = {m for part in split for m in part}
    per_rank = defaultdict(int)
    for s, part in enumerate(split):
        for m in part:
            per_rank[s % PP] += per_module.get(m, 0)
    for k in per_module:
        if k not in placed:
            per_rank[0 if "vision" in k else (n - 1) % PP] += per_module[k]
    _static[dim] = {r: 16 * per_rank[r] / GiB for r in range(PP)}
    return _static[dim]


def measured(cell):
    out = {}
    for r in range(PP):
        data = json.load(open(os.path.join(res, cell, "mem", f"rank{r}.json")))
        out[r] = data["records"][step]["max_allocated_gib"]
    return out


lo, hi = measured(cell_lo), measured(cell_hi)
s_lo, s_hi = static(dim_lo), static(dim_hi)
fit = {}
for r in range(PP):
    b = ((hi[r] - s_hi[r]) - (lo[r] - s_lo[r])) / (dim_hi - dim_lo)
    a = (lo[r] - s_lo[r]) - b * dim_lo
    fit[r] = (a, b)
print("measured at step %d: %s D=%d %s | %s D=%d %s" % (
    step, cell_lo, dim_lo, " ".join(f"{lo[r]:.2f}" for r in range(PP)),
    cell_hi, dim_hi, " ".join(f"{hi[r]:.2f}" for r in range(PP))))


def predict(dim):
    s = static(dim)
    return {r: s[r] + fit[r][0] + fit[r][1] * dim for r in range(PP)}


best = None
dim = 1024
while dim <= 16384:
    p = predict(dim)
    if max(p.values()) > target:
        break
    best = dim
    dim += 512
for d in sorted({best - 512, best, best + 512} | set(extra_dims)):
    p = predict(d)
    tag = "  <- chosen" if d == best else ""
    print(f"D={d}: predicted peak per rank " + " ".join(f"r{r} {p[r]:.1f}" for r in range(PP))
          + f" | max {max(p.values()):.1f} GiB; static " + " ".join(f"{static(d)[r]:.1f}" for r in range(PP)) + tag)
