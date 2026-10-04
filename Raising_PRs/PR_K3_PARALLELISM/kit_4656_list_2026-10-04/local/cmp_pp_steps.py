"""Compare per-rank step lines (loss, grad norm) of several PP run logs: the first log is the reference."""
import re, sys

PAT = re.compile(r"\[rank(\d+)\]:.*?step:\s*(\d+)\s+loss:\s*([-\d.e]+)\s+grad_norm:\s*([-\d.e]+)")

def read(path):
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="replace").read())
    return {(int(r), int(s)): (l, g) for r, s, l, g in PAT.findall(text)}

logs = sys.argv[1:]
ref = read(logs[0])
ranks = sorted({r for r, _ in ref})
steps = sorted({s for _, s in ref})
print(f"reference {logs[0]}: ranks {ranks}, steps {steps[0] if steps else None}..{steps[-1] if steps else None}")
for s in (1, 5, 10):
    vals = sorted({ref[k] for k in ref if k[1] == s})
    print(f"  step {s}: loss / grad_norm {vals}")
for path in logs[1:]:
    other = read(path)
    diff = [k for k in sorted(set(ref) | set(other)) if ref.get(k) != other.get(k)]
    print(f"{path}: {len(ref)} reference lines, {len(other)} lines, {len(diff)} differ" + (f", first {diff[0]}: {ref.get(diff[0])} vs {other.get(diff[0])}" if diff else ""))
