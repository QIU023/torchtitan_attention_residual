"""Summarize m4656_5033: per cell the step lines identical to main's, loss and grad norm at steps 1/10/20, the largest
relative loss gap to main, and the peak memory over steps after the first. usage: summarize_1012.py <runs dir>"""
import pathlib
import re
import sys

ROOT = pathlib.Path(sys.argv[1])
PAT = re.compile(r"\[rank(\d+)\]:.*?step:\s*(\d+)\s+loss:\s*([-\d.e]+)\s+grad_norm:\s*([-\d.e]+)\s+memory:\s*([\d.]+)GiB")


def read(name):
    p = ROOT / name / "run.log"
    if not p.exists():
        return None
    text = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text(errors="replace"))
    return {(int(r), int(s)): (l, g, float(m)) for r, s, l, g, m in PAT.findall(text)}


def compare(prefix, cell, others):
    ref = read(f"{prefix}_main_{cell}")
    if ref is None:
        return [f"== {prefix} {cell}: no main run"]
    out = [f"== {prefix} {cell}"]
    for t in others:
        got = read(f"{prefix}_{t}_{cell}")
        if got is None:
            out.append(f"  {t}: missing")
            continue
        same = sum(1 for k in ref if got.get(k, (None,))[:2] == ref[k][:2])
        gaps = [abs(float(got[k][0]) - float(ref[k][0])) / abs(float(ref[k][0])) for k in ref if k in got]
        peak = max((v[2] for (r, s), v in got.items() if s > 1), default=float("nan"))
        first_diff = min((s for (r, s) in ref if (r, s) in got and got[(r, s)][:2] != ref[(r, s)][:2]), default=None)
        out.append(f"  {t}: identical to main {same}/{len(ref)} step lines, first differing step {first_diff}, "
                   f"max relative loss gap {max(gaps, default=float('nan')):.2e}, peak {peak:.2f} GiB")
        for s in (1, 10, 20):
            rows = sorted({got[k][:2] for k in got if k[1] == s})
            if rows:
                out.append(f"    step {s}: {rows}")
    return out


if __name__ == "__main__":
    names = [d.name for d in ROOT.iterdir() if d.is_dir()]
    for prefix in ("num", "mem8192", "mem16384", "pp"):
        cells = sorted({n.split("_", 2)[2] for n in names if n.startswith(f"{prefix}_main_")})
        for cell in cells:
            print("\n".join(compare(prefix, cell, ("main", "main2", "maineager", "c1", "c2"))))
    if (ROOT / "pp_main").exists():
        ref = read("pp_main")
        for t in ("main2", "c1", "c2"):
            got = read(f"pp_{t}")
            if got is None or ref is None:
                continue
            same = sum(1 for k in ref if got.get(k, (None,))[:2] == ref[k][:2])
            peak = max((v[2] for (r, s), v in got.items() if s > 1), default=float("nan"))
            print(f"== pp {t}: identical to main {same}/{len(ref)} step lines, peak {peak:.2f} GiB (main {max(v[2] for (r, s), v in ref.items() if s > 1):.2f})")
