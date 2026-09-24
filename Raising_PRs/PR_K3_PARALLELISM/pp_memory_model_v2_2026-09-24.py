"""Per PP rank block-residual memory over one Interleaved1F1B step, with torch's runtime as measured.

Run from a checkout of pp_review4 (for the layout tables) with torch nightly:
    PYTHONPATH=<pp_review4>:. python pp_memory_model_v2_2026-09-24.py

Differences from pp_memory_timeline_2026-09-24.py, each measured on 8 x RTX 5060 on 09-24:
  - torch's _PipelineScheduleRuntime waits every send Work at the end of the step, and an NCCL
    Work pins the sent tensor (and the whole storage of a view) until waited; so every forward
    output, payload and input gradient a rank sends lives until the end of the step.
  - torch keeps one receive buffer per micro-batch for every input and every output gradient,
    hidden included, across steps.
Units: one block or hidden state of one micro-batch, tokens x dim x 2 bytes.

Schemes:
  base   pp_review4 901ef34de as implemented.
  v4     pp_review_optimize: receives straight into one rank-store buffer per micro-batch,
         stacks and payloads are views of it, all receive buffers on demand, forward sends
         waited at the stage's backward of the micro-batch.
  lower  the report's bound: each block once per rank from its arrival or commit to the rank's
         last backward of the micro-batch, nothing pinned past its use.
"""

import argparse
from collections import defaultdict

from torch.distributed.pipelining.schedules import _ComputationType, ScheduleInterleaved1F1B

from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part
from torchtitan.models.kimi_k3.pipeline_parallel.layout import (
    infer_block_layout_tables,
    layer_to_stage_from_split,
)


def schedule(pp, vp, m):
    s = object.__new__(ScheduleInterleaved1F1B)
    s.pp_group_size, s.n_local_stages, s._n_microbatches = pp, vp, m
    s.number_of_rounds = max(1, m // pp)
    s.microbatches_per_round = m // s.number_of_rounds
    order = {r: s._calculate_single_rank_operations(r) for r in range(pp)}
    width = max(len(v) for v in order.values())
    return {r: v + [None] * (width - len(v)) for r, v in order.items()}, width


def build(pp, vp, n_layers, block):
    S = pp * vp
    split = _generate_llm_fqn_per_model_part(S, n_layers, 1, 1)
    l2s = layer_to_stage_from_split(split)
    t = infer_block_layout_tables(
        stage_to_rank={s: s % pp for s in range(S)},
        n_layers=n_layers,
        layers_per_block=block,
        layer_to_stage=l2s,
    )
    layers = defaultdict(list)
    for layer, s in l2s.items():
        layers[s].append(layer)
    return t, layers


def simulate(pp, vp, m, n_layers, block, scheme, act_per_layer=1.0, fix=()):
    """Per rank: (persistent units, dynamic units per slot, layer-activation units per slot).

    fix removes one v4 gap at a time: "grad_pin" (input gradients sent are freed once used),
    "per_block" (the store keeps each block from its arrival to the rank's last backward),
    "hidden_out" (a forward output is freed once sent), "open_copy" (a block opened inside a
    stage is appended without copying the stack).
    """
    t, layers = build(pp, vp, n_layers, block)
    order, width = schedule(pp, vp, m)
    S = t.num_stages
    stages_of = {r: [s for s in range(S) if s % pp == r] for r in range(pp)}
    k_out = {s: len(t.delta_to_send(s)) if s < S - 1 else 0 for s in range(S)}
    k_in = {s: k_out[s - 1] if s else 0 for s in range(S)}
    n_in = {s: len(t.cache_at_entry(s)) + k_in[s] for s in range(S)}
    commits = {s: len(t.commits_at(s)) for s in range(S)}
    # block b opens at layer block * b; the new stack is saved by the stage's later layers, if any
    held_to_backward = {
        s: any(block * b < max(layers[s]) for b in t.commits_at(s)) for s in range(S)
    }
    rows_needed = {r: max(n_in[s] + commits[s] for s in stages_of[r]) for r in range(pp)}
    slot_f, slot_b = {}, {}
    for r in range(pp):
        for time, a in enumerate(order[r]):
            if a is None:
                continue
            key = (a.stage_index, a.microbatch_index)
            if a.computation_type == _ComputationType.FORWARD:
                slot_f[key] = time
            else:
                slot_b[key] = time
    end = width

    persistent = {r: 0 for r in range(pp)}
    if scheme == "base":
        for r in range(pp):
            for s in stages_of[r]:
                if s > 0:
                    persistent[r] += m * (1 + k_in[s])
                if s < S - 1:
                    persistent[r] += m * (1 + k_out[s])

    dyn = {r: [0.0] * width for r in range(pp)}
    acts = {r: [0.0] * width for r in range(pp)}

    def add(r, start, stop, units):
        for time in range(start, min(stop, width)):
            dyn[r][time] += units

    for r in range(pp):
        for mb in range(m):
            f_first = min(slot_f[(s, mb)] for s in stages_of[r])
            f_last = max(slot_f[(s, mb)] for s in stages_of[r])
            b_last = max(slot_b[(s, mb)] for s in stages_of[r])
            if scheme == "v4" and "per_block" not in fix:
                add(r, f_first, b_last + 1, rows_needed[r])
            if scheme == "v4" and "per_block" in fix:
                for blk in range(t.num_blocks):
                    arrivals = [
                        slot_f[(q, mb)] for q in stages_of[r]
                        if blk in t.commits_at(q) or (q and blk in t.delta_to_send(q - 1))
                    ]
                    if arrivals:
                        add(r, min(arrivals), b_last + 1, 1)
            for s in stages_of[r]:
                f, b = slot_f[(s, mb)], slot_b[(s, mb)]
                saved_inputs = len(layers[s]) - (1 if s else 0)  # a stage's first input is the hidden
                for time in range(f, b + 1):
                    acts[r][time] += act_per_layer * saved_inputs
                if scheme == "base":
                    if s > 0 and n_in[s]:
                        add(r, f, b + 1, n_in[s])  # the leaf stack copy
                    if commits[s]:
                        stop = b + 1 if held_to_backward[s] else f_last + 1
                        add(r, f, stop, n_in[s] + commits[s])  # the model's new stack
                    if s < S - 1:
                        add(r, f, end, 1 + k_out[s])  # hidden output and payload copy, pinned
                    if s > 0:
                        add(r, b, end, 1 + k_in[s])  # input gradients sent, pinned
                elif scheme == "v4":
                    if s > 0:
                        add(r, f, b + 1, 1)  # hidden input, received on demand
                    if commits[s] and held_to_backward[s] and "open_copy" not in fix:
                        add(r, f, b + 1, n_in[s] + commits[s])
                    if s < S - 1 and "hidden_out" not in fix:
                        add(r, f, b + 1, 1)  # hidden output until its send is waited
                    if s > 0 and "grad_pin" not in fix:
                        add(r, b, end, 1 + k_in[s])  # input gradients sent, pinned
                else:  # lower
                    if s > 0:
                        add(r, f, b + 1, 1)
                # deposits: from the first reader's backward to the backward of the stage that brought it
                for blk in t.cache_at_entry(s):
                    bringer = next(
                        q for q in stages_of[r]
                        if blk in t.commits_at(q) or (q and blk in t.delta_to_send(q - 1))
                    )
                    add(r, b, slot_b[(bringer, mb)] + 1, 1)
            if scheme == "lower":
                for blk in range(t.num_blocks):
                    arrivals = [
                        slot_f[(q, mb)] for q in stages_of[r]
                        if blk in t.commits_at(q) or (q and blk in t.delta_to_send(q - 1))
                    ]
                    if arrivals:
                        add(r, min(arrivals), b_last + 1, 1)
    return persistent, dyn, acts, width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pp", type=int, default=8)
    ap.add_argument("--vp", type=int, default=2)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--block", type=int, default=4)
    ap.add_argument("--unit-gib", type=float, default=3584 * 2048 * 2 / 2**30)
    ap.add_argument("--act", type=float, default=1.0)
    args = ap.parse_args()
    res = {}
    for scheme in ("base", "v4", "lower"):
        res[scheme] = simulate(args.pp, args.vp, args.m, args.layers, args.block, scheme, args.act)
    print(f"pp{args.pp} x vp{args.vp}, M={args.m}, {args.layers} layers in blocks of {args.block}, "
          f"unit {args.unit_gib:.4f} GiB, {args.act} unit(s) per layer beyond a stage's first")
    print("| rank | " + " | ".join(f"{s}: persistent + peak(dyn + acts) units" for s in res) +
          " | base - v4 GiB | v4 - lower GiB |")
    print("|---:|" + "---|" * len(res) + "---:|---:|")
    for r in range(args.pp):
        peaks = {}
        cells = []
        for scheme, (per, dyn, acts, width) in res.items():
            tot = [dyn[r][i] + acts[r][i] for i in range(width)]
            peaks[scheme] = per[r] + max(tot)
            cells.append(f"{per[r]} + {max(tot):.0f} = {peaks[scheme]:.0f}")
        print(f"| {r} | " + " | ".join(cells) +
              f" | {(peaks['base'] - peaks['v4']) * args.unit_gib:.2f} | {(peaks['v4'] - peaks['lower']) * args.unit_gib:.2f} |")
    gaps(args.pp, args.vp, args.m, args.layers, args.block, args.unit_gib)



def k3_layouts():
    """The 2.8T flavor (93 layers in blocks of 12, 8K tokens x 7168, TP1 CP1) on the feasible layouts."""
    unit = 8192 * 7168 * 2 / 2**30
    for label, pp, vp, m in (
        ("H100 PP8 x VP4", 8, 4, 16),
        ("H100 PP16 x VP2", 16, 2, 32),
        ("GB300 PP4 x VP4", 4, 4, 16),
        ("GB300 PP2 x VP8", 2, 8, 16),
    ):
        print(f"\n{label}, M={m}: block residual + hidden memory per rank, GiB (no layer activations)")
        print("| rank | base persistent + dynamic | v4 | lower | base / lower |")
        print("|---:|---|---:|---:|---:|")
        res = {s: simulate(pp, vp, m, 93, 12, s, 0.0) for s in ("base", "v4", "lower")}
        for r in range(pp):
            peak = {s: res[s][0][r] + max(res[s][1][r]) for s in res}
            print(f"| {r} | {res['base'][0][r] * unit:.1f} + {max(res['base'][1][r]) * unit:.1f} = {peak['base'] * unit:.1f} "
                  f"| {peak['v4'] * unit:.1f} | {peak['lower'] * unit:.1f} | {peak['base'] / peak['lower']:.1f} |")
        gaps(pp, vp, m, 93, 12, unit)


def gaps(pp, vp, m, n_layers, block, unit):
    """The v4 to lower-bound gap of the heaviest rank, one fix at a time and all together."""
    def worst(scheme, fix=()):
        per, dyn, _, _ = simulate(pp, vp, m, n_layers, block, scheme, 0.0, fix)
        return max(per[r] + max(dyn[r]) for r in range(pp)) * unit
    v4 = worst("v4")
    cells = [f"v4 {v4:.1f}"]
    for fix in ("grad_pin", "per_block", "hidden_out", "open_copy"):
        cells.append(f"-{fix} {v4 - worst('v4', (fix,)):.1f}")
    all_fix = worst("v4", ("grad_pin", "per_block", "hidden_out", "open_copy"))
    cells.append(f"all fixes {all_fix:.1f}")
    cells.append(f"lower {worst('lower'):.1f}")
    print("heaviest rank, GiB: " + ", ".join(cells))


if __name__ == "__main__":
    import sys

    if "--k3" in sys.argv:
        k3_layouts()
    else:
        main()
