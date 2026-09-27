"""Per rank and step: how many parameter gradients are bitwise equal between two dumps, tower vs text."""
import glob, os, sys, torch
a, b = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(a, "rank*_step*.pt"))):
    name = os.path.basename(f)
    x = torch.load(f); y = torch.load(os.path.join(b, name))
    keys = sorted(set(x) | set(y))
    rows = {"tower": [0, 0], "text": [0, 0]}
    worst = {"tower": (0.0, ""), "text": (0.0, "")}
    for k in keys:
        kind = "tower" if "vision_encoder" in k else "text"
        rows[kind][1] += 1
        if k in x and k in y and torch.equal(x[k], y[k]):
            rows[kind][0] += 1
        elif k in x and k in y:
            d = ((x[k].float() - y[k].float()).norm() / max(x[k].float().norm().item(), 1e-30)).item()
            if d > worst[kind][0]:
                worst[kind] = (d, k)
    print(name, " ".join(f"{kind} equal {r[0]}/{r[1]}" for kind, r in rows.items()),
          " ".join(f"| {kind} max rel {w[0]:.2e} {w[1]}" for kind, w in worst.items() if w[1]))
