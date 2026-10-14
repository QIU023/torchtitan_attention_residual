"""Per layout: r1 (cold cache) vs r1b (warm copy) on steps 1-10, and r1b vs r2 (resumed, warm copy) on steps 6-10."""
import pathlib, re, sys
O = pathlib.Path(sys.argv[1])
PAT = re.compile(r"step:\s*(\d+)\s+loss:\s*([-\d.]+)\s+grad_norm:\s*([-\d.]+)")
def read(d):
    t = re.sub(r"\x1b\[[0-9;]*m", "", (d / "run.log").read_text(errors="replace"))
    out = {}
    for s, l, g in PAT.findall(t):
        if not l.startswith("-"):
            out.setdefault(int(s), (l, g))
    return out
for r1 in sorted(O.glob("r1_*")):
    name = r1.name[3:]
    if not (O / f"r1b_{name}").exists():
        continue
    a, ab, b = read(r1), read(O / f"r1b_{name}"), read(O / f"r2_{name}")
    cold = [s for s in range(1, 11) if a.get(s) == ab.get(s) and s in a]
    resumed = [s for s in range(6, 11) if ab.get(s) == b.get(s) and s in b]
    print(f"{name:16s} r1 vs r1b identical {len(cold)}/10; r1b vs r2 identical {len(resumed)}/5",
          "" if len(resumed) == 5 else f"diffs {[(s, ab.get(s), b.get(s)) for s in range(6, 11) if ab.get(s) != b.get(s)]}")
