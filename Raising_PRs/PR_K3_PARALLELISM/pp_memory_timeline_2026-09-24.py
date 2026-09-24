"""Per PP rank memory over one step of Interleaved1F1B, for Kimi K3's block residual.

Run from a checkout of pp_review4 (for the layout tables) with torch nightly:
    PYTHONPATH=<pp_review4>:. python pp_memory_timeline_2026-09-24.py

Units: one block B = tokens x dim x 2 bytes (bf16) for one micro-batch; one layer
activation a = what one layer saves for backward for one micro-batch. The schedule is
torch's own ScheduleInterleaved1F1B order (time-aligned, one slot per F or B).

Block accounting per scheme, per rank and slot:
  ours          as pp_review4 implements it: each stage's forward stacks its N input blocks
                into a fresh leaf held until its backward; a committing stage's output stack
                (N + c) is held by the block's own later aggregations; the outgoing payload
                is a fresh stack held in fwd_cache until backward; the received delta stays
                in the rank store until the rank's last forward of the micro-batch; backward
                deposits stay until the stage that brought the block collects them.
  lower bound   every block a rank uses held once, from its arrival or commit on the rank
                until the rank's last backward of the micro-batch (the K3 report's claim).
  hybrid        ours, with jinsooihm's rule: the adjacent payload carries own + other-parity
                blocks; odd offsets from 3 are direct sends that sit in the receiver's store
                from the producer's forward on.
  +offload      the rank store parks its blocks on host (the k3_pp_offload draft): received
                deltas leave the device right after the forward that stored them.
"""

import sys
from collections import defaultdict

from torch.distributed.pipelining.schedules import _ComputationType, ScheduleInterleaved1F1B

from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part
from torchtitan.models.kimi_k3.pipeline_parallel.layout import (
    infer_block_layout_tables,
    layer_to_stage_from_split,
)

N_LAYERS, BLOCK = 93, 12


def schedule(pp, vp, m):
    s = object.__new__(ScheduleInterleaved1F1B)
    s.pp_group_size, s.n_local_stages, s._n_microbatches = pp, vp, m
    s.number_of_rounds = max(1, m // pp)
    s.microbatches_per_round = m // s.number_of_rounds
    order = {r: s._calculate_single_rank_operations(r) for r in range(pp)}
    width = max(len(v) for v in order.values())
    return {r: v + [None] * (width - len(v)) for r, v in order.items()}, width


def layout(pp, vp):
    S = pp * vp
    split = _generate_llm_fqn_per_model_part(S, N_LAYERS, 1, 1)
    l2s = layer_to_stage_from_split(split)
    t = infer_block_layout_tables(
        stage_to_rank={s: s % pp for s in range(S)},
        n_layers=N_LAYERS,
        layers_per_block=BLOCK,
        layer_to_stage=l2s,
    )
    layers = defaultdict(int)
    for layer, s in l2s.items():
        layers[s] += 1
    return t, layers


def hybrid_routes(t, pp):
    """Adjacent payload per hop and direct sends (producer stage, receiver stage, block)."""
    S = t.num_stages
    payload = {s: [] for s in range(S)}
    direct = []
    for b in range(t.num_blocks):
        pb = t.producer_stage_of_block(b)
        for d in range(1, pp):
            recv = pb + d
            if recv >= S:
                break
            if d == 1 or d % 2 == 0:
                payload[recv - 1].append(b)
            else:
                direct.append((pb, recv, b))
    return payload, direct


def is_fwd(a):
    return a.computation_type == _ComputationType.FORWARD


def block_timeline(pp, vp, m, scheme, offload=False):
    """Block memory per rank per slot, in B: (persistent, dynamic) lists.

    persistent: recv buffers torch's PipelineStage allocates once per micro-batch
    (_setup_forward_recv_info / _prepare_backward_infra): the delta each stage receives
    and the gradient of the payload it sends; the hybrid's direct sends are given the
    same per-micro-batch buffers. The lower bound receives straight into its store.
    """
    t, _ = layout(pp, vp)
    order, width = schedule(pp, vp, m)
    S = t.num_stages
    _, l2s = layout_split(pp, vp)
    stages_of = {r: [s for s in range(S) if s % pp == r] for r in range(pp)}
    n_in = {s: len(t.cache_at_entry(s)) + (len(t.delta_to_send(s - 1)) if s else 0) for s in range(S)}
    commits = {s: len(t.commits_at(s)) for s in range(S)}
    # a committed block's new stack is held by the layers after its opening layer, if any
    held_to_backward = {
        s: any(l2s.get(BLOCK * b + 1) == s for b in t.commits_at(s)) for s in range(S)
    }
    direct_in = defaultdict(list)
    direct_out = defaultdict(int)
    if scheme in ("hybrid", "hybrid_dyn", "hybrid_lower"):
        payload, direct = hybrid_routes(t, pp)
        k_out = {s: len(payload[s]) for s in range(S)}
        for pb, recv, b in direct:
            direct_in[recv].append((pb, b))
            direct_out[pb] += 1
    else:
        k_out = {s: len(t.delta_to_send(s)) for s in range(S)}
    k_in = {s: (k_out[s - 1] if s else 0) for s in range(S)}

    f_time = {}
    b_time = {}
    last_b_on_rank = {}
    for r in range(pp):
        for time, a in enumerate(order[r]):
            if a is None:
                continue
            if is_fwd(a):
                f_time[(a.stage_index, a.microbatch_index)] = time
            else:
                b_time[(a.stage_index, a.microbatch_index)] = time
                last_b_on_rank[(r, a.microbatch_index)] = time

    persistent = {r: 0.0 for r in range(pp)}
    if scheme in ("ours", "hybrid", "hybrid_dyn"):
        for r in range(pp):
            for s in stages_of[r]:
                persistent[r] += m * (k_in[s] + (k_out[s] if s < S - 1 else 0))
                if scheme == "hybrid":
                    persistent[r] += m * (len(direct_in[s]) + direct_out[s])

    dyn = {r: [0.0] * width for r in range(pp)}
    for r in range(pp):
        live = defaultdict(float)
        for time in range(width):
            a = order[r][time]
            if scheme == "hybrid_lower" and not offload:
                for s in stages_of[r]:
                    for pb, b in direct_in.get(s, ()):
                        for mb in range(m):
                            if f_time.get((pb, mb)) == time:
                                live[("blk", mb, b)] = 1.0
            if scheme == "hybrid_dyn":
                # a direct block sits on the receiver from the producer's forward to the
                # receiver rank's release; its gradient sits on the producer from the
                # receiver stage's backward to the producer stage's backward
                for s_recv in stages_of[r]:
                    for pb, b in direct_in.get(s_recv, ()):
                        for mb in range(m):
                            if f_time.get((pb, mb)) == time:
                                live[("direct", s_recv, mb, b)] = 1.0
                for s_prod in stages_of[r]:
                    for (pb, recv_stage, b) in [(pb, rs, b) for rs, lst in direct_in.items() for pb, b in lst if pb == s_prod]:
                        for mb in range(m):
                            if b_time.get((recv_stage, mb)) == time:
                                live[("dgrad", s_prod, mb, b)] = 1.0
            if scheme == "ours_dyn":
                for s_send in stages_of[r]:
                    if s_send + 1 < S:
                        for mb in range(m):
                            if b_time.get((s_send + 1, mb)) == time and k_out[s_send]:
                                live[("pgrad", s_send, mb)] = k_out[s_send]
            if a is not None:
                s, mb = a.stage_index, a.microbatch_index
                if scheme == "ours_dyn" and is_fwd(a) and s > 0 and k_in[s]:
                    live[("rbuf", s, mb)] = k_in[s]
                if scheme == "ours_dyn" and is_fwd(a) and s == max(stages_of[r]):
                    for key in [k for k in live if k[0] == "rbuf" and k[2] == mb]:
                        del live[key]
                if scheme == "ours_dyn" and not is_fwd(a):
                    live.pop(("pgrad", s, mb), None)
                if scheme in ("ours", "ours_dyn", "hybrid", "hybrid_dyn"):
                    if is_fwd(a):
                        live[("leaf", s, mb)] = n_in[s]
                        live[("payload", s, mb)] = k_out[s] if s < S - 1 else 0
                        if commits[s] and (held_to_backward[s] or not offload):
                            live[("out", s, mb)] = n_in[s] + commits[s]
                        if s == max(stages_of[r]):          # the store lets go of mb
                            for key in [k for k in live if k[0] == "out" and k[2] == mb
                                        and not held_to_backward[k[1]]]:
                                del live[key]
                            for key in [k for k in live if k[0] == "direct" and k[2] == mb]:
                                del live[key]
                    else:
                        for kind in ("leaf", "out", "payload"):
                            live.pop((kind, s, mb), None)
                        for key in [k for k in live if k[0] == "dgrad" and k[1] == s and k[2] == mb]:
                            del live[key]
                        for b in t.cache_at_entry(s):
                            live[("dep", mb, b)] = 1.0
                        brought = list(t.commits_at(s)) + (t.delta_to_send(s - 1) if s else [])
                        for b in brought:
                            live.pop(("dep", mb, b), None)
                else:  # lower bounds
                    if is_fwd(a):
                        used = set(t.cache_at_entry(s)) | set(t.commits_at(s))
                        used |= set(t.delta_to_send(s - 1)) if s else set()
                        if not offload:
                            for b in used:
                                live[("blk", mb, b)] = 1.0
                    else:
                        for b in t.cache_at_entry(s):
                            live[("dep", mb, b)] = 1.0
                        brought = list(t.commits_at(s)) + (t.delta_to_send(s - 1) if s else [])
                        for b in brought:
                            live.pop(("dep", mb, b), None)
                        if time == last_b_on_rank[(r, mb)]:
                            for key in [k for k in live if k[0] == "blk" and k[1] == mb]:
                                del live[key]
            dyn[r][time] = sum(live.values())
    return persistent, dyn


def layout_split(pp, vp):
    S = pp * vp
    split = _generate_llm_fqn_per_model_part(S, N_LAYERS, 1, 1)
    return split, layer_to_stage_from_split(split)


def activation_timeline(pp, vp, m):
    """Layer-activation units a resident per rank per slot (layers of each stage in flight)."""
    _, layers = layout(pp, vp)
    order, width = schedule(pp, vp, m)
    out = {r: [0] * width for r in range(pp)}
    for r in range(pp):
        cur = 0
        for time, a in enumerate(order[r]):
            if a is not None:
                cur += layers[a.stage_index] * (1 if is_fwd(a) else -1)
            out[r][time] = cur
    return out, order


def spark(values, peak, cols=64):
    bars = " ▁▂▃▄▅▆▇█"
    n = len(values)
    out = []
    for c in range(cols):
        seg = values[c * n // cols:(c + 1) * n // cols] or [0]
        v = max(seg)
        out.append(bars[min(8, round(8 * v / peak))] if peak else " ")
    return "".join(out)


# Kimi-K3 in-tree flavor, measured on meta: per-layer non-expert and expert params (B)
MLA_LAYERS = set(range(3, 92, 4)) | {92}
EXPERTS_PER_MOE_LAYER = 29.60
EMBED, HEAD, VISION = 1.17, 1.17, 0.45
UNIT_GB = 8192 * 7168 * 2 / 2**30           # one block (or hidden) for an 8K-token micro-batch


def nonexpert(layer):
    if layer == 0:
        return 1.170
    return 0.422 if layer in MLA_LAYERS else 0.634


def static_gb(pp, vp, ep, dp, grad_buffer_gb=2.0):
    """bf16 params + ZeRO-1 fp32 master and momentum shards; grads live on CPU (Pipeline ZeRO-2)."""
    _, l2s = layout_split(pp, vp)
    S = pp * vp
    out = {}
    for r in range(pp):
        layers = [l for l, st in l2s.items() if st % pp == r]
        experts = sum(EXPERTS_PER_MOE_LAYER / ep for l in layers if l)
        dense = sum(nonexpert(l) for l in layers)
        dense += (EMBED + VISION) * (r == 0) + HEAD * (r == (S - 1) % pp)
        params = 2 * (experts + dense)
        optim = 8 * experts / dp + 8 * dense / (dp * ep)
        out[r] = dict(layers=len(layers), params=params, optim=optim, grad=grad_buffer_gb,
                      total=params + optim + grad_buffer_gb)
    return out


def hidden_buffers(pp, vp, m):
    S = pp * vp
    return {r: m * sum((s > 0) + (s < S - 1) for s in range(S) if s % pp == r) for r in range(pp)}


def balance(totals, movable):
    """Move a fraction of each rank's movable memory to the lightest rank so the peaks meet."""
    ranks = list(totals)
    dest = min(ranks, key=lambda r: max(totals[r]))
    width = len(totals[dest])
    lo, hi = max(totals[dest]), max(max(v) for v in totals.values())
    for _ in range(60):
        x = (lo + hi) / 2
        frac = {r: 0.0 if r == dest or max(totals[r]) <= x else
                min(1.0, (max(totals[r]) - x) / max(movable[r])) for r in ranks}
        dest_line = [totals[dest][t] + sum(frac[r] * movable[r][t] for r in ranks) for t in range(width)]
        if max(dest_line) > x:
            lo = x
        else:
            hi = x
    x = hi
    frac = {r: 0.0 if r == dest or max(totals[r]) <= x else
            min(1.0, (max(totals[r]) - x) / max(movable[r])) for r in ranks}
    out = {r: [totals[r][t] - frac[r] * movable[r][t] for t in range(width)] for r in ranks}
    out[dest] = [totals[dest][t] + sum(frac[r] * movable[r][t] for r in ranks) for t in range(width)]
    return out, dest, frac


def report(pp, vp, m, ep, dp, act_units_per_layer, label):
    S = pp * vp
    t, _ = layout(pp, vp)
    acts, order = activation_timeline(pp, vp, m)
    width = len(order[0])
    stat = static_gb(pp, vp, ep, dp)
    hid = hidden_buffers(pp, vp, m)
    print(f"\n=== {label}: pp{pp} x vp{vp}, ep{ep}, dp{dp}, {m} micro-batches of 8K tokens, "
          f"{S} stages, {t.num_blocks} blocks; 1 unit = {UNIT_GB:.3f} GiB; a = {act_units_per_layer} unit(s) per layer")
    print("| rank | layers | static GiB | layer-activations peak (layer x mb) | torch hidden recv buffers (units) |")
    print("|---:|---:|---:|---:|---:|")
    for r in range(pp):
        print(f"| {r} | {stat[r]['layers']} | {stat[r]['total']:.1f} | {max(acts[r])} | {hid[r]} |")
    schemes = [("ours", False), ("ours", True), ("ours_dyn", False), ("hybrid", False), ("hybrid_dyn", False),
               ("lower", False), ("lower", True)]
    blocks = {}
    print("\n| scheme | " + " | ".join(f"rank {r} persistent + dynamic peak (units)" for r in range(pp)) + " |")
    print("|---|" + "---:|" * pp)
    for scheme, off in schemes:
        per, dyn = block_timeline(pp, vp, m, scheme, off)
        blocks[(scheme, off)] = {r: [per[r] + v for v in dyn[r]] for r in range(pp)}
        name = {"ours": "this PR", "ours_dyn": "this PR, on-demand buffers", "hybrid": "hybrid, per-mb buffers", "hybrid_dyn": "hybrid, on-demand buffers", "lower": "lower bound"}[scheme] + (" + offload" if off else "")
        print(f"| {name} | " + " | ".join(f"{per[r]:.0f} + {max(dyn[r]):.0f} = {max(blocks[(scheme, off)][r]):.0f}" for r in range(pp)) + " |")
    lines = {}
    for key, blk in blocks.items():
        totals = {r: [stat[r]['total'] + UNIT_GB * (hid[r] + blk[r][i] + act_units_per_layer * acts[r][i])
                      for i in range(width)] for r in range(pp)}
        movable = {r: [UNIT_GB * act_units_per_layer * acts[r][i] for i in range(width)] for r in range(pp)}
        bal, dest, frac = balance(totals, movable)
        lines[key] = (totals, bal, dest, frac)
    print("\n| scheme | " + " | ".join(f"rank {r} peak GiB" for r in range(pp)) + " | with balance (per rank) |")
    print("|---|" + "---:|" * pp + "---|")
    for key, (totals, bal, dest, frac) in lines.items():
        name = {"ours": "this PR", "ours_dyn": "this PR, on-demand buffers", "hybrid": "hybrid, per-mb buffers", "hybrid_dyn": "hybrid, on-demand buffers", "lower": "lower bound"}[key[0]] + (" + offload" if key[1] else "")
        print(f"| {name} | " + " | ".join(f"{max(totals[r]):.1f}" for r in range(pp)) +
              " | " + ", ".join(f"{max(bal[r]):.1f}" for r in range(pp)) + f" (pool on rank {dest}) |")
    return lines, width


if __name__ == "__main__":
    for pp, vp, m, dp, label in ((8, 4, 16, 16, "H100 80 GB"), (4, 8, 8, 16, "GB300 288 GB")):
        for a in (1, 17):
            lines, width = report(pp, vp, m, 64, dp, a, label)
            if a == 1:
                keys = (("ours", False), ("hybrid_dyn", False), ("lower", False))
                peak = max(max(max(v) for v in lines[k][0].values()) for k in keys)
                for key in keys:
                    totals, bal, dest, _ = lines[key]
                    print(f"--- {key[0]} total, a=1 (scale {peak:.0f} GiB); then with balance")
                    for r in range(pp):
                        print(f"  rank {r}: {spark(totals[r], peak)}  {max(totals[r]):.1f}")
                    for r in range(pp):
                        print(f"  bal  {r}: {spark(bal[r], peak)}  {max(bal[r]):.1f}")
