"""Per layout, all on warm copies of one cache: r1b vs r1c on steps 1-10 (determinism), r1b vs r2b on steps 6-10 (resume)."""
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
    name = r1b.name[4:]
    a, c, b = read(r1b), read(O / f"r1c_{name}"), read(O / f"r2b_{name}")
    det = [s for s in range(1, 11) if s in a and a.get(s) == c.get(s)]
    res = [s for s in range(6, 11) if s in b and a.get(s) == b.get(s)]
    print(f"{name:16s} r1b vs r1c identical {len(det)}/10; r1b vs r2b identical {len(res)}/5; r2b starts at {min(b) if b else None}",
          "" if len(res) == 5 else f"diffs {[(s, a.get(s), b.get(s)) for s in range(6, 11) if a.get(s) != b.get(s)]}")
