"""Per-class relative-difference distribution of step-1 gradient dumps against dp1 (rank-0 files; sharded parameters are gathered by the dump hack)."""
import collections, statistics, sys, torch
G, ref, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
a = torch.load(f"{G}/{ref}.rank0.pt", map_location="cpu")
def rel(x, y): x = x.double(); y = y.double(); return ((x - y).norm() / max(x.norm().item(), 1e-30)).item()
def group(k):
    k = k.replace("._checkpoint_wrapped_module", "")
    for key, g in (("vision_encoder", "vision"), ("tok_embeddings", "embedding"), ("lm_head", "head"), ("output_res", "output_res"), ("routed_experts", "experts"), ("router", "router"), ("shared_experts", "shared"), ("routed_", "moe seams"), ("delta_attention", "kda"), ("attention_res", "res"), ("ffn_res", "res"), ("attention", "mla"), ("feed_forward", "ffn"), ("norm", "norms")):
        if key in k: return g
    return "other"
for tag in tags:
    try: b = torch.load(f"{G}/{tag}.rank0.pt", map_location="cpu")
    except Exception as e: print(f"### {tag}: missing ({type(e).__name__})"); continue
    rels = []; by = collections.defaultdict(list)
    for k, g in a.items():
        if k not in b or b[k].shape != g.shape: continue
        e = rel(g, b[k]); rels.append(e); by[group(k)].append(e)
    rs = sorted(rels); q = lambda p: rs[min(len(rs) - 1, int(p * len(rs)))]
    print(f"### {tag} vs {ref}: {len(rels)} params; median {statistics.median(rels):.1e} p90 {q(0.9):.1e} max {max(rels):.1e}; within 1e-4: {sum(r <= 1e-4 for r in rels)}, 1e-3: {sum(r <= 1e-3 for r in rels)}, 1e-2: {sum(r <= 1e-2 for r in rels)}")
    print("    " + "  ".join(f"{g}:{statistics.median(v):.0e}/{max(v):.0e}" for g, v in sorted(by.items(), key=lambda kv: -max(kv[1]))))
