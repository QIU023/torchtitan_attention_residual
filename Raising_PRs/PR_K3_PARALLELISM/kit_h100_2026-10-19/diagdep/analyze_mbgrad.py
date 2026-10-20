"""Rebuild the step-1 tower gradient from the replicas' per-micro-batch gradients two ways and compare with the dumps:
fp32 sum then one bf16 rounding (DEP's buffer), and a bf16 sum in stage 0's backward order (an accumulating .grad).
Run with the tree's own torch, which also gives the Interleaved1F1B order. Usage: <G> <tree>"""
import sys
from pathlib import Path

import torch
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B

G, tree = Path(sys.argv[1]), sys.argv[2]


def stage0_backward_order(pp=4, vp=4, m=16):
    num_stages = pp * vp
    s = ScheduleInterleaved1F1B.__new__(ScheduleInterleaved1F1B)
    s._num_stages, s.pp_group_size, s._n_microbatches, s.n_microbatches = num_stages, pp, m, m
    s.stage_index_to_group_rank = {i: i % pp for i in range(num_stages)}
    s.number_of_rounds = max(1, m // pp)
    s.microbatches_per_round = m // s.number_of_rounds

    class _Stage:
        def __init__(self, i):
            self.stage_index, self.num_stages, self.group_rank = i, num_stages, i % pp
            self.is_first, self.is_last = i == 0, i == num_stages - 1

    s._stages = [_Stage(i) for i in range(0, num_stages, pp)]
    s.n_local_stages, s.rank = len(s._stages), 0
    acts = [a for a in s._calculate_single_rank_operations(0) if a is not None]
    return [a.microbatch_index for a in acts if a.stage_index == 0 and "BACKWARD" in a.computation_type.name]


mb = {}
for r in range(4):
    mb.update(torch.load(G / f"mb_{tree}" / "dump" / f"mbgrad_rank{r}.pt"))
names = sorted({n for _, n in mb})
mbs = sorted({m for m, _ in mb})
order = stage0_backward_order()
print(f"{tree}: torch {torch.__version__}; replica gradients for micro-batches {mbs}; stage-0 backward order {order}")
on = torch.load(G / f"mb_{tree}" / "dump" / "grad1_rank0.pt")
off = torch.load(G / f"{tree}_w_dep_off" / "dump" / "grad1_rank0.pt")
prev_on = torch.load(G / f"{tree}_w_dep_k25" / "dump" / "grad1_rank0.pt")


def owner(name, d):
    hits = [k for k in d if k.endswith(name)]
    assert len(hits) == 1, (name, hits[:3])
    return d[hits[0]]


counts = {"fp32 -> DEP on": 0, "fp32 -> DEP off": 0, "bf16 in order -> DEP off": 0, "bf16 in order -> DEP on": 0, "this DEP on == earlier DEP on": 0}
for name in names:
    parts = [mb[(m, name)] for m in mbs if (m, name) in mb]
    fp32 = torch.stack([p.float() for p in parts]).sum(0).bfloat16()
    acc = None
    for m in order:
        if (m, name) in mb:
            g = mb[(m, name)]
            acc = g.clone() if acc is None else (acc.float() + g.float()).bfloat16()
    counts["fp32 -> DEP on"] += torch.equal(fp32, owner(name, on))
    counts["fp32 -> DEP off"] += torch.equal(fp32, owner(name, off))
    counts["bf16 in order -> DEP off"] += torch.equal(acc, owner(name, off))
    counts["bf16 in order -> DEP on"] += torch.equal(acc, owner(name, on))
    counts["this DEP on == earlier DEP on"] += torch.equal(owner(name, on), owner(name, prev_on))
for k, v in counts.items():
    print(f"   {k}: {v}/{len(names)}")
