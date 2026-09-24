"""Size the PP block-memory probe so pp_review4 fills about 72 GiB of an 80 GB H100 per rank on 4 GPUs.

Run from a checkout of pp_review4 with torch nightly:
    PYTHONPATH=<kit>:<pp_review4>:. python h100_probe_sizing.py

Peak per rank = static + unit x (block, hidden and layer-input units from the calibrated model)
+ transient. Static is 16 bytes per parameter on the rank (fp32 master, fp32 grad, two AdamW
states); the probe's parameters are counted on meta. The transient (the backward recompute of
one layer, dominated by the AttnRes aggregation's fp32 copies) is fitted on the 8 x 5060 V4 run
as a + b x (the rank's largest stack + 1) units, plus the head on the last rank; the fit is
checked against the 5060 baseline before it is used.
"""

import importlib.util
import json
import os
from collections import defaultdict

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "memv2", os.path.join(HERE, "..", "pp_memory_model_v2_2026-09-24.py")
)
memv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(memv2)

GiB = 2**30


_PARAMS: dict = {}


def probe_params(dim, layers, block, seq):
    """Parameters per top-level module of the probe flavor, counted on meta (independent of seq)."""
    key = (dim, layers, block)
    if key in _PARAMS:
        return _PARAMS[key]
    os.environ.update(
        PPMEM_DIM=str(dim), PPMEM_LAYERS=str(layers), PPMEM_BLOCK=str(block),
        PPMEM_SEQ=str(min(seq, 16384)),
    )
    import probe_ppmem

    config = probe_ppmem.ppmem_wide()
    with torch.device("meta"):
        model = config.model.build()
    out = defaultdict(int)
    for name, p in model.named_parameters():
        parts = name.split(".")
        module = ".".join(parts[:2]) if parts[0] == "layers" else parts[0]
        out[module] += p.numel()
    _PARAMS[key] = out
    return out


def static_gib(pp, vp, layers, params):
    S = pp * vp
    split = memv2._generate_llm_fqn_per_model_part(S, layers, 1, 1)
    stage_layers = defaultdict(list)
    for layer, s in memv2.layer_to_stage_from_split(split).items():
        stage_layers[s].append(layer)
    per_rank = defaultdict(int)
    for s in range(S):
        r = s % pp
        for layer in stage_layers[s]:
            per_rank[r] += params[f"layers.{layer}"]
        if s == 0:
            per_rank[r] += params.get("tok_embeddings", 0) + params.get("vision_encoder", 0)
        if s == S - 1:
            for key in ("norm", "lm_head", "output_res_proj", "output_res_norm"):
                per_rank[r] += params.get(key, 0)
    return {r: 16 * n / GiB for r, n in per_rank.items()}


def max_stack(pp, vp, layers, block):
    t, _ = memv2.build(pp, vp, layers, block)
    S = t.num_stages
    return {
        r: max(
            len(t.cache_at_entry(s)) + (len(t.delta_to_send(s - 1)) if s else 0) + len(t.commits_at(s))
            for s in range(S) if s % pp == r
        )
        for r in range(pp)
    }


def model_units(pp, vp, m, layers, block, scheme):
    per, dyn, acts, width = memv2.simulate(pp, vp, m, layers, block, scheme, 1.0)
    return {r: per[r] + max(dyn[r][i] + acts[r][i] for i in range(width)) for r in range(pp)}


def fit_transient():
    """a + b (N + 1) units per rank, plus a head term on the last rank, from the 5060 V4 run."""
    unit = 3584 * 2048 * 2 / GiB
    params = probe_params(2048, 32, 4, 3584)
    stat = static_gib(8, 2, 32, params)
    nmax = max_stack(8, 2, 32, 4)
    units = {s: model_units(8, 2, 16, 32, 4, s) for s in ("base", "v4")}
    peaks = {}
    for run in ("base2_s3584", "v4_s3584"):
        peaks[run] = {
            r: json.load(open(os.path.join(HERE, "results", run, f"rank{r}.json")))["records"][3]["max_allocated_gib"]
            for r in range(8)
        }
    xs, ys = [], []
    for r in range(7):  # the last rank carries the head; fit it separately
        xs.append(nmax[r] + 1)
        ys.append((peaks["v4_s3584"][r] - stat[r]) / unit - units["v4"][r])
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    head = (peaks["v4_s3584"][7] - stat[7]) / unit - units["v4"][7] - (a + b * (nmax[7] + 1))
    print(f"transient fit on the 5060 V4 run: {a:.0f} + {b:.1f} x (N + 1) units, head +{head:.0f} units")
    print("check on the 5060 baseline (GiB): rank: measured / predicted")
    for r in range(8):
        tr = a + b * (nmax[r] + 1) + (head if r == 7 else 0)
        pred = stat[r] + unit * (units["base"][r] + tr)
        print(f"  r{r}: {peaks['base2_s3584'][r]:.2f} / {pred:.2f}")
    return a, b, head


def predict(pp, vp, m, layers, block, dim, seq, fit, scheme):
    a, b, head = fit
    unit = seq * dim * 2 / GiB
    params = probe_params(dim, layers, block, seq)
    stat = static_gib(pp, vp, layers, params)
    nmax = max_stack(pp, vp, layers, block)
    units = model_units(pp, vp, m, layers, block, scheme)
    S = pp * vp
    last = (S - 1) % pp
    return {
        r: stat[r] + unit * (units[r] + a + b * (nmax[r] + 1) + (head if r == last else 0))
        for r in range(pp)
    }, sum(params.values())


def search(pp, vp, m, layers, block, fit, target, *, dim=None, seq=None):
    """Grow seq (dim fixed) or dim (seq fixed) until pp_review4's heaviest rank passes target."""
    step = 512 if dim else 256
    lo = 1024 if dim else 1024
    best = None
    x = lo
    while x <= 65536:
        d, t = (dim, x) if dim else (x, seq)
        peak, nparams = predict(pp, vp, m, layers, block, d, t, fit, "base")
        if max(peak.values()) > target:
            break
        best = (d, t, peak, nparams)
        x += step
    d, t, base_peak, nparams = best
    v4_peak, _ = predict(pp, vp, m, layers, block, d, t, fit, "v4")
    # V4 capacity: the longest micro-batch at the same dim that stays under target
    t4 = t
    while True:
        peak, _ = predict(pp, vp, m, layers, block, d, t4 + 512, fit, "v4")
        if max(peak.values()) > target:
            break
        t4 += 512
    return d, t, base_peak, v4_peak, nparams, t4


def main():
    fit = fit_transient()
    target = 72.0
    print(f"\nH100 x 4 (80 GB), 16 attention heads, M = 16; the probe is sized so pp_review4's heaviest rank "
          f"reaches {target} GiB allocated")
    print("| layout | layers / block | block opens mid-stage | dim | seq | params | pp_review4 max rank | V4 same config | "
          "V4 longest micro-batch under the same peak |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    # 16 stages on 4 ranks need titan's layers_per_stage to divide layers + 2: 46 layers at 3 per stage,
    # 30 at 2; both open every block inside a stage, as the released layout does.
    for pp, vp, m, layers, block in ((4, 4, 16, 46, 6), (4, 4, 16, 30, 4)):
        t_, sl = memv2.build(pp, vp, layers, block)
        mid = sum(any(block * b < max(sl[s]) for b in t_.commits_at(s)) for s in range(t_.num_stages))
        for mode in ("seq8192", "dim2048"):
            if mode == "seq8192":
                d, t, base_peak, v4_peak, nparams, t4 = search(pp, vp, m, layers, block, fit, target, seq=8192)
            else:
                d, t, base_peak, v4_peak, nparams, t4 = search(pp, vp, m, layers, block, fit, target, dim=2048)
            print(f"| pp{pp} x vp{vp} | {layers} / {block} | {mid} of {t_.num_stages} | {d} | {t} | {nparams / 1e6:.0f} M "
                  f"| {max(base_peak.values()):.1f} | {max(v4_peak.values()):.1f} | {t4} ({t4 / t:.2f}x) |")
            print("    per rank pp_review4: " + ", ".join(f"{v:.1f}" for v in base_peak.values())
                  + "; V4: " + ", ".join(f"{v:.1f}" for v in v4_peak.values()))


if __name__ == "__main__":
    main()
