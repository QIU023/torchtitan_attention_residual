"""Bitwise comparison of two ab_pipeline.py output dirs.

usage: python ab_compare.py <dir_a> <dir_b>
"""

import os
import sys

import torch

a_dir, b_dir = sys.argv[1], sys.argv[2]
names = sorted(f for f in os.listdir(a_dir) if f.endswith(".pt"))
bad = 0
tensors = 0
for f in names:
    a = torch.load(os.path.join(a_dir, f), weights_only=False)
    b = torch.load(os.path.join(b_dir, f), weights_only=False)
    if a["plan"] != b["plan"]:
        bad += 1
        print(f, "plan differs")
    for step, ((ga, la), (gb, lb)) in enumerate(zip(a["history"], b["history"], strict=True)):
        if ga.keys() != gb.keys():
            bad += 1
            print(f, step, "grad keys differ")
        for k in ga:
            x, y = ga[k], gb[k]
            if (x is None) != (y is None) or (x is not None and not torch.equal(x, y)):
                bad += 1
                print(f, step, "grad", k, "differs")
            tensors += x is not None
        for x, y in zip(la, lb, strict=True):
            tensors += 1
            if not torch.equal(x, y):
                bad += 1
                print(f, step, "loss differs", x, y)
    for step, (ea, eb) in enumerate(zip(a["evals"], b["evals"], strict=True)):
        for x, y in zip(ea, eb, strict=True):
            tensors += 1
            if not torch.equal(x, y):
                bad += 1
                print(f, step, "eval loss differs")
print(f"files {len(names)}, tensors compared {tensors}, differences {bad}")
sys.exit(1 if bad else 0)
