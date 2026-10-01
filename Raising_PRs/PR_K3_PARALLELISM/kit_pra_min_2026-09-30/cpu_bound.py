"""CPU block accounting for the Kimi K3 PP stage (local only): four gloo ranks, pp4 x vpp2,
Interleaved1F1B, a toy model whose blocks are separate tensors. After every stage action, the unique
storages held by the rank store, the stages' cached block inputs and outputs, and forward sends still
alive, against the tight bound (a block freed at the backward of the stage that brought it).

usage: python cpu_bound.py <tree> <stack|list> [<tree> <stack|list> ...]
"""

import json
import os
import sys
import tempfile
import weakref

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

T, D, LPB, NUM_LAYERS, MB, WORLD = 8, 64, 2, 16, 8, 4
SPLIT = [[2 * s, 2 * s + 1] for s in range(8)]
BLOCK_BYTES = T * D * 4


class Toy(nn.Module):
    def __init__(self, layers, first, last, stack_mode):
        super().__init__()
        self.layers, self.first, self.last, self.stack_mode = layers, first, last, stack_mode
        self.w = nn.Parameter(torch.ones(D))

    def forward(self, hidden, blocks=None):
        if self.stack_mode and blocks is not None:
            blocks = list(blocks.unbind(1))
        if self.first or blocks is None:
            blocks = []
        for layer in self.layers:
            if layer % LPB == 0:
                blocks = [*blocks, hidden * self.w]
            hidden = hidden + 0.01 * torch.stack(blocks, 0).sum(0)
        if self.last:
            return (hidden.sum() + torch.stack(blocks, 0).sum()).reshape(1)
        if self.stack_mode:
            return hidden, torch.stack(blocks, 1) if blocks else hidden.new_zeros(T, 0, D)
        return hidden, blocks


def _storages(tensors):
    out = {}
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.numel():
            st = t.untyped_storage()
            out[st.data_ptr()] = st.nbytes()
    return out


def _flat(x):
    if isinstance(x, torch.Tensor):
        yield x
    elif isinstance(x, dict):
        for v in x.values():
            yield from _flat(v)
    elif isinstance(x, (list, tuple)):
        for v in x:
            yield from _flat(v)


def worker(rank, tree, stack_mode, store_file, out_file):
    sys.path.insert(0, tree)
    from torch.distributed.pipelining.schedules import _PipelineScheduleRuntime, ScheduleInterleaved1F1B

    from torchtitan.models.kimi_k3.pipeline_parallel.cache import PPRankLocalCache
    from torchtitan.models.kimi_k3.pipeline_parallel.layout import infer_block_layout_tables
    from torchtitan.models.kimi_k3.pipeline_parallel.stage import AttnResPipelineStage

    from torch.distributed.pipelining import PipelineStage

    base_send = PipelineStage.get_fwd_send_ops

    def tracked_send(self, chunk, *a, **k):
        ops = base_send(self, chunk, *a, **k)
        if recording[0]:
            for op in list(ops)[int(os.environ.get("SKIP_HIDDEN", "1")):]:
                t = getattr(op, "tensor", None)
                if isinstance(t, torch.Tensor):
                    sent.append(weakref.ref(t))
        return ops

    PipelineStage.get_fwd_send_ops = tracked_send
    sent, records, tight, recording = [], [], {}, [False]
    dist.init_process_group("gloo", init_method=f"file:///{store_file}", rank=rank, world_size=WORLD)
    torch.manual_seed(0)
    mine = list(range(rank, len(SPLIT), WORLD))
    last = len(SPLIT) - 1
    mods = [Toy(SPLIT[s], s == 0, s == last, stack_mode) for s in mine]
    stages = [AttnResPipelineStage(m, s, len(SPLIT), torch.device("cpu")) for m, s in zip(mods, mine)]
    schedule = ScheduleInterleaved1F1B(list(stages), n_microbatches=MB, loss_fn=lambda o, t: o.sum(), scale_grads=False)
    layout = infer_block_layout_tables(
        stage_to_rank=dict(stages[0].stage_index_to_group_rank),
        n_layers=NUM_LAYERS,
        layers_per_block=LPB,
        layer_to_stage={l: s for s, ls in enumerate(SPLIT) for l in ls},
        cache=True,
    )
    store = PPRankLocalCache()
    runtime = isinstance(schedule, _PipelineScheduleRuntime)
    for st in stages:
        try:
            st.set_routing(layout, store, wait_sends_at_backward=runtime)
        except TypeError:
            st.set_routing(layout, store)


    def brought(st):
        s = st.stage_index
        return (layout.delta_to_send(s - 1) if s else []) + layout.commits_at(s)

    def account():
        blocks = _storages(_flat(store._blocks))
        for st in stages:
            for out_tuple, ins in st.fwd_cache.values():
                if not st.is_first:
                    blocks.update(_storages(list(ins)[1:]))
                if not st.is_last:
                    blocks.update(_storages(list(out_tuple)[1:]))
        blocks.update(_storages(r() for r in sent if r() is not None))
        return sum(blocks.values())

    for st in stages:
        fwd, bwd = st.forward_one_chunk, st.backward_one_chunk

        def f(chunk, *a, _st=st, _fwd=fwd, **k):
            out = _fwd(chunk, *a, **k)
            for b in brought(_st):
                tight[(chunk, b)] = BLOCK_BYTES
            if recording[0]:
                records.append((f"{_st.stage_index}F{chunk}", account(), sum(tight.values())))
            return out

        def b_(chunk, *a, _st=st, _bwd=bwd, **k):
            out = _bwd(chunk, *a, **k)
            for b in brought(_st):
                tight.pop((chunk, b), None)
            if recording[0]:
                records.append((f"{_st.stage_index}B{chunk}", account(), sum(tight.values())))
            return out

        st.forward_one_chunk, st.backward_one_chunk = f, b_

    x = torch.randn(MB * T, D)
    for step in range(2):
        recording[0] = step == 1
        sent.clear()
        if 0 in mine:
            schedule.step(x)
        elif last in mine:
            schedule.step(target=torch.zeros(MB))
        else:
            schedule.step()
    peak = max(records, key=lambda r: r[1])
    worst = max(r[1] - r[2] for r in records)
    res = {"rank": rank, "peak_action": peak[0], "peak_bytes": peak[1], "tight_at_peak": peak[2],
           "max_tight": max(r[2] for r in records), "max_excess": worst}
    with open(f"{out_file}.{rank}", "w") as fh:
        json.dump(res, fh)
    dist.destroy_process_group()


if __name__ == "__main__":
    args = sys.argv[1:]
    for tree, mode in zip(args[0::2], args[1::2]):
        d = tempfile.mkdtemp()
        out = os.path.join(d, "res")
        mp.spawn(worker, args=(tree, mode == "stack", os.path.join(d, "store"), out), nprocs=WORLD, join=True)
        rows = [json.load(open(f"{out}.{r}")) for r in range(WORLD)]
        print(f"== {os.path.basename(tree)} ({mode})")
        for r in rows:
            print(f"  rank {r['rank']}: peak {r['peak_bytes'] / BLOCK_BYTES:5.1f} blocks at {r['peak_action']:>5} "
                  f"(tight there {r['tight_at_peak'] / BLOCK_BYTES:4.1f}, tight max {r['max_tight'] / BLOCK_BYTES:4.1f}), "
                  f"max excess over tight {r['max_excess'] / BLOCK_BYTES:4.1f} blocks")
