"""For the Kimi K3 debug model (24 layers, blocks of 12) under a pipeline shape, print what each rank caches and the
order in which each block's gradient contributions are added in the backward, with the rank cache on and off,
following pipeline_stage.py (collect: grad_col.add_(deposit); deposit: prior + grad; the owner collects before its
backward). l<s> = stage s's own contribution (its layers' AttnRes reads of the block, plus the head's aggregation on
the last stage), combined with what arrived from the next stage by the stage's own autograd, the same in both modes.
usage: PYTHONPATH=<tree> python pp_cache_reduction_order.py <pp> <stages per rank>"""
import sys

from torchtitan.models.kimi_k3.layout import BlockLayoutTables, layer_to_stage_from_split
from torchtitan.models.kimi_k3.parallelize import kimi_k3_module_fqns_per_model_part

pp, vp = int(sys.argv[1]), int(sys.argv[2])
S, N_LAYERS, BLOCK = pp * vp, 24, 12
split = kimi_k3_module_fqns_per_model_part(S, N_LAYERS)
l2s = layer_to_stage_from_split(split)
rank = {s: s % pp for s in range(S)}
name = {0: "e", 1: "x12"}
tabs = {c: BlockLayoutTables(stage_to_rank=rank, num_blocks=2, n_layers=N_LAYERS, layers_per_block=BLOCK,
                             layer_to_stage=l2s, cache=c) for c in (True, False)}
on = tabs[True]

print(f"### pp{pp} x vp{vp}: {S} stages, stage s on rank s % {pp}\n")
print("| stage (rank) | layers | commits | cache at entry (on) | P2P out, cache on | P2P out, cache off |")
print("| --- | --- | --- | --- | --- | --- |")
for s in range(S):
    layers = [n.split(".")[1] for n in split[s] if n.startswith("layers.")]
    lay = f"{layers[0]}-{layers[-1]}" if len(layers) > 1 else (layers[0] if layers else "-")
    extra = [n for n in split[s] if not n.startswith("layers.")]
    if "tok_embeddings" in extra: lay = "emb, " + lay
    if "lm_head" in extra: lay = lay + ", head" if lay != "-" else "head"
    f = lambda bs: "[" + ", ".join(name[b] for b in bs) + "]"
    print(f"| s{s} (r{rank[s]}) | {lay} | {f(on.commits_at(s)) if on.commits_at(s) else '-'} | {f(sorted(on.cache_at_entry(s))) if on.cache_at_entry(s) else '-'} "
          f"| {f(on.delta_to_send(s)) if s < S - 1 else '-'} | {f(tabs[False].delta_to_send(s)) if s < S - 1 else '-'} |")


print("\nPer rank, cache on (which stage brought the block onto the rank, which later stages there read it from the store):\n")
for r in range(pp):
    parts = []
    for b in (0, 1):
        stages = [s for s in range(S) if rank[s] == r]
        readers = [s for s in on.cache_readers_of_block(b) if rank[s] == r]
        holders = [s for s in stages if b in on.commits_at(s) or (s > 0 and b in on.delta_to_send(s - 1))]
        if holders or readers:
            h = holders[0] if holders else None
            how = "commits" if h is not None and b in on.commits_at(h) else "receives"
            parts.append(f"`{name[b]}` ({how} at s{h}; store readers {', '.join('s%d' % x for x in readers) or 'none'})")
    print(f"- r{r}: " + ("; ".join(parts) if parts else "nothing cached"))

def tree(tab, b):
    owner = tab.producer_stage_of_block(b)
    recv, dep = {}, {}
    for s in range(S - 1, owner - 1, -1):
        r = rank[s]
        if s == owner:
            g = recv.get(s)
            if dep.get(r):
                g = f"({g} + {dep[r]})" if g else dep[r]
            return f"(l{s} + {g})" if g else f"l{s}"
        col = f"(l{s} + {recv[s]})" if s in recv else f"l{s}"
        arrived = b in tab.delta_to_send(s - 1)
        if arrived:
            if dep.get(r):
                col = f"({col} + {dep.pop(r)})"
            recv[s - 1] = col
        else:
            dep[r] = f"({dep[r]} + {col})" if dep.get(r) else col
    raise AssertionError


print()
for b in (0, 1):
    o = on.producer_stage_of_block(b)
    print(f"- block `{name[b]}` (committed at s{o}):")
    print(f"  - cache off: `{tree(tabs[False], b)}`")
    print(f"  - cache on:  `{tree(on, b)}`")
