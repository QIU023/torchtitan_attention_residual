"""Every PP rank's peak with and without pp_balance (the k3_pp_balance draft), for pp_review4, V4 and the bound.

Run from a checkout of pp_review4 with torch nightly:
    PYTHONPATH=<kit>:<pp_review4>:. python balance_overlay.py

Per rank and slot: static + non-movable units + movable units + the backward transient. Movable is
what pp_balance can park: tensors autograd saves that are contiguous and held by nothing else (the
layer inputs FullAC saves beyond a stage's first layer, and the stack a block opening leaves for
the stage's later layers once nothing but checkpointing holds it). The draft's rules: one
destination rank preallocates a pool of P GiB for the whole step, split evenly among the source
ranks; a source parks while its span has room (it saves min(span, movable) at every slot) and
keeps a 256 MiB staging buffer. The search picks the destination, the sources (the heaviest k
ranks) and P that minimise the largest peak over all ranks.
"""

import h100_probe_sizing as hs

memv2 = hs.memv2
GiB = hs.GiB
STAGING = 0.25


def timelines(pp, vp, m, layers, block, dim, seq, fit, scheme, act=1.0):
    """Per rank: (total GiB per slot, movable GiB per slot)."""
    a, b, head = fit
    unit = seq * dim * 2 / GiB
    params = hs.probe_params(dim, layers, block, seq)
    stat = hs.static_gib(pp, vp, layers, params)
    nmax = hs.max_stack(pp, vp, layers, block)
    mov = {}
    per, dyn, acts, width = memv2.simulate(pp, vp, m, layers, block, scheme, act, movable_out=mov)
    last = (pp * vp - 1) % pp
    out = {}
    for r in range(pp):
        transient = unit * (a + b * (nmax[r] + 1) + (head if r == last else 0))
        movable = [unit * (acts[r][i] + mov[r][i]) for i in range(width)]
        total = [stat[r] + transient + unit * (per[r] + dyn[r][i]) + movable[i] for i in range(width)]
        out[r] = (total, movable)
    return out


def balance(lines):
    pp = len(lines)
    unbalanced = {r: max(lines[r][0]) for r in range(pp)}
    best = (max(unbalanced.values()), None, (), 0.0, unbalanced)
    order = sorted(range(pp), key=lambda r: -unbalanced[r])
    for dest in range(pp):
        others = [r for r in order if r != dest]
        for k in range(1, len(others) + 1):
            sources = others[:k]
            pool = 0.25
            while pool <= 40:
                span = pool / k
                peaks = {}
                for r in range(pp):
                    total, movable = lines[r]
                    if r == dest:
                        peaks[r] = max(total) + pool
                    elif r in sources:
                        peaks[r] = max(t - min(span, mv) for t, mv in zip(total, movable)) + STAGING
                    else:
                        peaks[r] = max(total)
                worst = max(peaks.values())
                if worst < best[0] - 1e-9:
                    best = (worst, dest, tuple(sources), pool, peaks)
                pool += 0.25
    return best


def report(label, pp, vp, m, layers, block, dim, seq, fit, measured=None, act=1.0):
    print(f"\n{label}: pp{pp} x vp{vp}, M={m}, {layers} layers in blocks of {block}, dim {dim}, seq {seq}, "
          f"{act} unit(s) saved per layer beyond a stage's first")
    head = "| scheme | " + " | ".join(f"r{r}" for r in range(pp)) + " | max | mean | with pp_balance: per rank | max | knobs |"
    print(head)
    print("|---|" + "---:|" * pp + "---:|---:|---|---:|---|")
    if measured:
        for name, vals in measured.items():
            print(f"| {name} (measured) | " + " | ".join(f"{vals[r]:.2f}" for r in range(pp)) +
                  f" | {max(vals.values()):.2f} | {sum(vals.values()) / pp:.2f} | | | |")
    results = {}
    for scheme in ("base", "v4", "lower"):
        lines = timelines(pp, vp, m, layers, block, dim, seq, fit, scheme, act)
        un = {r: max(lines[r][0]) for r in range(pp)}
        worst, dest, sources, pool, peaks = balance(lines)
        movable_at_peak = {r: max(lines[r][1]) for r in range(pp)}
        knobs = "none helps" if dest is None else f"dest r{dest}, sources {list(sources)}, pool {pool:.2f} GiB"
        print(f"| {scheme} | " + " | ".join(f"{un[r]:.2f}" for r in range(pp)) +
              f" | {max(un.values()):.2f} | {sum(un.values()) / pp:.2f} | " +
              ", ".join(f"{peaks[r]:.1f}" for r in range(pp)) + f" | {worst:.2f} | {knobs} |")
        print("    largest movable GiB per rank: " + ", ".join(f"{movable_at_peak[r]:.2f}" for r in range(pp)))
        results[scheme] = (un, worst)
    return results


def main():
    import json
    import os

    fit = hs.fit_transient()
    here = os.path.dirname(os.path.abspath(__file__))
    measured = {}
    for run, name in (("base2_s3584", "base"), ("v4_s3584", "v4")):
        measured[name] = {
            r: json.load(open(os.path.join(here, "results", run, f"rank{r}.json")))["records"][3]["max_allocated_gib"]
            for r in range(8)
        }
    report("8 x RTX 5060 probe", 8, 2, 16, 32, 4, 2048, 3584, fit, measured)
    report("4 x H100 probe, FullAC", 4, 4, 16, 46, 6, 4096, 8192, fit)
    for act in (4.0, 8.0):
        report("4 x H100 probe, more saved per layer (a selective AC)", 4, 4, 16, 46, 6, 4096, 8192, fit, act=act)




def size_balanced(pp, vp, m, layers, block, fit, target=72.0, seq=8192, act=1.0):
    """The dim at which pp_review4 with pp_balance reaches target on its heaviest rank, then V4 there."""
    dim, found = 2048, None
    while dim <= 8192:
        worst, dest, sources, pool, peaks = balance(timelines(pp, vp, m, layers, block, dim, seq, fit, "base", act))
        if worst > target:
            break
        found = (dim, worst, peaks, dest, sources, pool)
        dim += 128
    dim, base_worst, base_peaks, dest, sources, pool = found
    v4 = balance(timelines(pp, vp, m, layers, block, dim, seq, fit, "v4", act))
    seq4 = seq
    while True:
        w = balance(timelines(pp, vp, m, layers, block, dim, seq4 + 512, fit, "v4", act))[0]
        if w > target:
            break
        seq4 += 512
    print(f"\nSized with pp_balance on both trees ({act} unit(s) per layer): dim {dim}, seq {seq}")
    print(f"  pp_review4 + balance per rank: " + ", ".join(f"{v:.1f}" for v in base_peaks.values()) +
          f" (max {base_worst:.1f}; dest r{dest}, sources {list(sources)}, pool {pool:.2f} GiB)")
    print(f"  V4 + balance per rank: " + ", ".join(f"{v:.1f}" for v in v4[4].values()) +
          f" (max {v4[0]:.1f}; dest r{v4[1]}, sources {list(v4[2])}, pool {v4[3]:.2f} GiB)")
    print(f"  V4 + balance longest micro-batch under {target} GiB: {seq4} tokens ({seq4 / seq:.2f}x)")


if __name__ == "__main__":
    import sys

    if "--size" not in sys.argv:
        main()
        raise SystemExit
    fit = hs.fit_transient()
    for act in (1.0, 4.0):
        size_balanced(4, 4, 16, 46, 6, fit, act=act)
