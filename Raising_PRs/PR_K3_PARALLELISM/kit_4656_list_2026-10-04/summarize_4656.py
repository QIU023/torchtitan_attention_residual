"""Summarize the m4656_list results: per cell rc, per-rank step lines identical or not, steps 1/10/50/100, peak memory."""
import pathlib, re, sys

ROOT = pathlib.Path(sys.argv[1])
PAT = re.compile(r"\[rank(\d+)\]:.*?step:\s*(\d+)\s+loss:\s*([-\d.e]+)\s+grad_norm:\s*([-\d.e]+)\s+memory:\s*([\d.]+)GiB")

def read(name):
    p = ROOT / name / "run.log"
    if not p.exists():
        return None
    text = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text(errors="replace"))
    return {(int(r), int(s)): (l, g, float(m)) for r, s, l, g, m in PAT.findall(text)}

cells = sorted({d.name.split("_", 1)[1] for d in ROOT.iterdir() if d.is_dir() and d.name.startswith("main_")})
for cell in cells:
    ref = read(f"main_{cell}")
    out = [f"== {cell}"]
    for t in ("main", "list", "main2"):
        got = read(f"{t}_{cell}")
        if got is None:
            continue
        steps = sorted({s for _, s in got})
        peak = max((v[2] for v in got.values()), default=float("nan"))
        same = sum(1 for k in ref if got.get(k, (None,))[:2] == ref[k][:2])
        out.append(f"  {t}: {len(got)} lines, steps {steps[0] if steps else None}..{steps[-1] if steps else None}, "
                   f"identical to main {same}/{len(ref)}, peak {peak:.2f} GiB")
    last = {}
    for (r, s), v in ref.items():
        last.setdefault(s, set()).add(v[:2])
    for s in (1, 10, 20, 50, 100):
        if s in last:
            out.append(f"  step {s}: {sorted(last[s])}")
    print("\n".join(out))
