"""Per arm: r1 (cold) vs r1b (warm copy) on steps 1-10, r1b vs r2b (resumed from r1b's step 5, warm copy) on steps 6-10."""
import pathlib, re, sys
O = pathlib.Path(sys.argv[1])
PAT = re.compile(r"step:\s*(\d+)\s+loss:\s*([-\d.]+)\s+grad_norm:\s*([-\d.]+)")
def read(d):
    p = d / "run.log"
    if not p.exists():
        return {}
    t = re.sub(r"\x1b\[[0-9;]*m", "", p.read_text(errors="replace"))
    out = {}
    for s, l, g in PAT.findall(t):
        if not l.startswith("-"):
            out.setdefault(int(s), (l, g))
    return out
for r1b in sorted(O.glob("r1b_*")):
    a = r1b.name[4:]
    x, y, z = read(O / f"r1_{a}"), read(r1b), read(O / f"r2b_{a}")
    cold = [s for s in range(1, 11) if s in x and x.get(s) == y.get(s)]
    res = [s for s in range(6, 11) if s in z and y.get(s) == z.get(s)]
    print(f"{a:10s} r1 vs r1b identical {len(cold)}/10; r1b vs r2b identical {len(res)}/5",
          "" if len(res) == 5 else f"diffs {[(s, y.get(s), z.get(s)) for s in range(6, 11) if y.get(s) != z.get(s)]}")
