"""How many initial parameters are bitwise equal between two dumps, per rank."""
import glob, os, sys, torch
a, b = sys.argv[1], sys.argv[2]
for f in sorted(glob.glob(os.path.join(a, "rank*.pt"))):
    x = torch.load(f); y = torch.load(os.path.join(b, os.path.basename(f)))
    keys = sorted(set(x) & set(y))
    equal = [k for k in keys if torch.equal(x[k], y[k])]
    diff = [k for k in keys if k not in equal]
    print(os.path.basename(f), f"equal {len(equal)}/{len(keys)}", "first differing:", diff[:3])
