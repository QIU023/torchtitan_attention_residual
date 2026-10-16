"""Stack check after the fold: logged loss and grad norm, the stage-0 tower's step-1 gradients compared between cells,
and its parameters after step 10 compared across the ranks that hold stage 0 (pp rank 0: dp x cp)."""
import re
import sys
from pathlib import Path

import torch

O = Path(sys.argv[1])
ansi = re.compile(r"\x1b\[[0-9;]*m")
pat = re.compile(r"step:\s*(\d+)\s+loss:\s*(-?[0-9.]+)\s+grad_norm:\s*([0-9.]+)")
mesh_pat = re.compile(r"STATE_DUMP grad1 rank 0 \d+ tensors tower mesh (.*)")


def series(name):
    out = {}
    text = ansi.sub("", (O / name / "run.log").read_text(errors="replace"))
    for line in text.splitlines():
        m = pat.search(line)
        if m and float(m.group(2)) >= 0:
            out.setdefault(int(m.group(1)), (m.group(2), m.group(3)))
    mesh = mesh_pat.search(text)
    return out, mesh.group(1) if mesh else "-"


def load(name, tag, r):
    return torch.load(O / name / "dump" / f"{tag}_rank{r}.pt")


def rel(a, b):
    worst, same = 0.0, 0
    for k in a:
        same += torch.equal(a[k], b[k])
        n = b[k].double().norm().item()
        d = (a[k].double() - b[k].double()).norm().item()
        worst = max(worst, d / n if n > 0 else d)
    return same, len(a), worst


for layout in ("c2", "c4"):
    print(f"===== {layout}")
    names = {(dep, dyn): f"{layout}_dep{dep}_{dyn}_meas" for dep in (0, 1) for dyn in ("off", "on")}
    for (dep, dyn), name in names.items():
        s, mesh = series(name)
        steps = sorted(s)
        print(f"{name:20s} steps {len(steps)} step1 {s[1]} step2 {s.get(2)} step10 {s.get(10)} tower mesh {mesh}")
    for (dep, dyn), name in names.items():
        ref = load(name, "param_last", 0)
        rows = [rel(load(name, "param_last", r), ref) for r in (1, 2, 3)]
        print(f"{name:20s} params after step 10, ranks 1-3 vs rank 0: " + ", ".join(f"{s}/{n} bitwise" for s, n, _ in rows))
    g = {key: load(name, "grad1", 0) for key, name in names.items()}
    for label, a, b in (
        ("DEP on vs off, dyn CP off (floor)", (1, "off"), (0, "off")),
        ("DEP on vs off, dyn CP on", (1, "on"), (0, "on")),
        ("dyn CP on vs off, DEP off", (0, "on"), (0, "off")),
        ("dyn CP on vs off, DEP on", (1, "on"), (1, "off")),
    ):
        s, n, w = rel(g[a], g[b])
        print(f"step-1 tower grads, {label:34s}: bitwise {s}/{n}, worst rel {w:.3e}")
