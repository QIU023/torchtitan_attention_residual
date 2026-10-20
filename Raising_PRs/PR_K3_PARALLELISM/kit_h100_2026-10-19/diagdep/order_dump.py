"""Interleaved1F1B order for pp4 x vpp4 x 16 micro-batches from the installed torch (the DEP plan test's helper)."""
import sys

import torch
from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B


def orders(pp=4, vp=4, m=16):
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

    out = {}
    for rank in range(pp):
        s._stages = [_Stage(i) for i in range(rank, num_stages, pp)]
        s.n_local_stages, s.rank = len(s._stages), rank
        out[rank] = " ".join("-" if a is None else str(a) for a in s._calculate_single_rank_operations(rank))
    return out


for rank, text in orders().items():
    print(f"rank {rank}: {text}")
