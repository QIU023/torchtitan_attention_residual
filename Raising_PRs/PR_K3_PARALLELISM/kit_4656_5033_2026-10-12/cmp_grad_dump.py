"""Compare two cells' step-1 gradient dumps (kit local/grad_dump) rank by rank: identical, or the relative norm gap."""
import glob
import sys

import torch

a_prefix, b_prefix = sys.argv[1], sys.argv[2]
for a_path in sorted(glob.glob(f"{a_prefix}.rank*.pt")):
    rank = a_path.rsplit(".rank", 1)[1].split(".")[0]
    a, b = torch.load(a_path), torch.load(f"{b_prefix}.rank{rank}.pt")
    same, diff = 0, []
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b:
            diff.append((float("inf"), name, "missing"))
            continue
        if torch.equal(a[name], b[name]):
            same += 1
        else:
            rel = ((a[name] - b[name]).norm() / a[name].norm().clamp_min(1e-30)).item()
            diff.append((rel, name, ""))
    print(f"rank {rank}: {same} identical, {len(diff)} differ")
    for rel, name, note in sorted(diff, reverse=True)[:8]:
        print(f"   {rel:.3e}  {name} {note}")
