"""Modelled DEP bubble fill on the Interleaved1F1B action order, in time units of one text-stage forward:
a text forward 1, a text backward 2 (1F1B's usual ratio; with full AC it is about 3, so the model understates
the cooldown bubbles), a slot as long as its longest action, a transfer 1. A micro-batch's ViT forward = r,
its recompute plus backward = 3r. Fill rate = ViT time placed in bubbles / all bubble time of the step."""
import importlib.util, sys
W, OLD = sys.argv[1], sys.argv[2]
sys.path.insert(0, W)
spec = importlib.util.spec_from_file_location("t", f"{W}/tests/unit_tests/cpu/test_kimi_k3_dep_plan.py")
t = importlib.util.module_from_spec(spec); sys.modules["t"] = t; spec.loader.exec_module(t)
spec = importlib.util.spec_from_file_location("old", OLD); old = importlib.util.module_from_spec(spec); sys.modules["old"] = old; spec.loader.exec_module(old)
from torchtitan.models.kimi_k3.pipeline_parallel import dep_plan as new

def run(pp, vp, m, r):
    order = t._interleaved_order(pp, vp, m)
    loads = {mb: 1024 for mb in range(m)}
    kw = dict(num_microbatches=m, num_ranks=pp, stage0_rank=0, trainable=True, pipeline_order=order, cost_ratio=r)
    b = new.plan_dep(loads, **kw)
    a = old.plan_dep(loads, **kw)
    runs, step_end, times = new._idle_runs(order)
    T = times[step_end]
    bubble = sum(x.stop - x.begin for rr in runs.values() for x in rr)
    fwd_in = sum(e - s for (k, (_, s, e)) in b.placed.items() if k[0] == "encode")
    bwd_in = sum(e - s for (k, (_, s, e)) in b.placed.items() if k[0] == "backward")
    text = m * pp * vp * 3.0
    vit = m * r * 4.0
    pro = max(r * len(v) for v in b.prologue.values())
    epi = max(3 * r * len(v) for v in b.epilogue.values())
    stalls = t._stalls(pp, vp, m, b, loads)
    return dict(T=T, bubble=bubble / (pp * T), fill=(fwd_in + bwd_in) / bubble if bubble else 0.0,
                share=vit / (vit + text), fwd=fwd_in / (m * r), bwd=bwd_in / (3 * m * r),
                nb_old=sum(1 for k in a.placed if k[0] == "backward"), nb=sum(1 for k in b.placed if k[0] == "backward"),
                exposed=(pro + epi) / T, inline=m * r * 4.0 / (pp * T) , stalls=len(stalls))

print("| layout | r | ViT share of compute | bubble share of the step | fill rate | ViT fwd hidden | ViT bwd hidden | backwards in bubbles, before -> after the fix | ViT left outside the schedule, share of the step |")
print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for pp, vp, m in ((2, 4, 4), (2, 4, 16), (4, 2, 8), (4, 2, 16), (4, 4, 16), (8, 2, 16), (8, 4, 16), (8, 4, 32)):
    for r in (0.5, 1.0):
        x = run(pp, vp, m, r)
        flag = "" if x["stalls"] == 0 else f" STALLS {x['stalls']}"
        print(f"| pp{pp} x vp{vp}, M{m} | {r} | {x['share']:.0%} | {x['bubble']:.0%} | {x['fill']:.0%} | {x['fwd']:.0%} | {x['bwd']:.0%} | {x['nb_old']} -> {x['nb']} of {m} | {x['exposed']:.1%} |{flag}")
