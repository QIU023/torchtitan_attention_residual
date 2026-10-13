"""Compare steps 6..10 of run 1 (continuous) and run 2 (resumed from step 5) per layout."""
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
    a, b = read(r1), read(O / f"r2_{name}")
    steps = [s for s in range(6, 11) if s in a and s in b]
    same = [s for s in steps if a[s] == b[s]]
    first = min(b) if b else None
    print(f"{name:12s} resumed run starts at step {first}; steps 6-10 identical {len(same)}/5", "" if len(same) == 5 else f"diffs {[(s, a.get(s), b.get(s)) for s in range(6, 11) if a.get(s) != b.get(s)]}")
