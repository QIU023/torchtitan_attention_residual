"""Compare a squashed stack commit (or several) with a PR's own diff, file by file, as multisets of +/- lines.

usage (inside the torchtitan checkout): python cmp_stack.py <name> <stack commit,...> "<PR base> <PR head>"
A line added and removed again within one side cancels, so code that only moved does not count as a difference.
"""
import collections
import subprocess
import sys


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout


def lines(diff, acc=None):
    out = acc if acc is not None else collections.defaultdict(collections.Counter)
    cur = None
    for l in diff.split("\n"):
        if l.startswith("diff --git"):
            cur = l.split(" b/")[-1]
            continue
        if l.startswith("+++") or l.startswith("---"):
            continue
        if cur and (l.startswith("+") or l.startswith("-")):
            out[cur][l] += 1
    return out


def cancel(side):
    for f in list(side):
        c = side[f]
        for l in list(c):
            if l.startswith("+") and ("-" + l[1:]) in c:
                k = min(c[l], c["-" + l[1:]])
                c[l] -= k
                c["-" + l[1:]] -= k
        side[f] = +c


name, stack_commits, pr_range = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
s = collections.defaultdict(collections.Counter)
for c in stack_commits:
    lines(git("show", "--format=", c), s)
p = lines(git("diff", *pr_range.split()))
cancel(s)
cancel(p)
print(f"### {name}: stack {stack_commits} vs PR [{pr_range}]")
for f in sorted(set(s) | set(p)):
    a, b = +s.get(f, collections.Counter()), +p.get(f, collections.Counter())
    if a == b:
        print(f"  same  {f} ({sum(a.values())} lines)")
        continue
    only_s, only_p = a - b, b - a
    print(f"  DIFF  {f}: stack-only {sum(only_s.values())}, PR-only {sum(only_p.values())}")
    if sum(only_s.values()) + sum(only_p.values()) <= 12:
        for l in list(only_s.elements())[:6]:
            print("        stack:", l[:150])
        for l in list(only_p.elements())[:6]:
            print("        PR:   ", l[:150])
