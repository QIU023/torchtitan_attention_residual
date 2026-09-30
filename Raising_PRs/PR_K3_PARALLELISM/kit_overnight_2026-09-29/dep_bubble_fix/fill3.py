"""Modelled DEP bubble fill at a fixed text : ViT compute ratio of the step, in time units of one text-stage
forward (text forward 1, text backward 2, a slot as long as its longest action, a transfer 1; a micro-batch's
ViT forward r, its recompute plus backward 3r). Fill rate = ViT time placed in bubbles / all bubble time."""
import importlib.util, sys
W, OLD = sys.argv[1], sys.argv[2]
sys.path.insert(0, W)
spec = importlib.util.spec_from_file_location("t", f"{W}/tests/unit_tests/cpu/test_kimi_k3_dep_plan.py")
t = importlib.util.module_from_spec(spec); sys.modules["t"] = t; spec.loader.exec_module(t)
spec = importlib.util.spec_from_file_location("old", OLD); old = importlib.util.module_from_spec(spec); sys.modules["old"] = old; spec.loader.exec_module(old)
from torchtitan.models.kimi_k3.pipeline_parallel import dep_plan as new

def run(pp, vp, m, share):
    text_mb = pp * vp * 3.0
    r = share / (1 - share) * text_mb / 4.0
    order = t._interleaved_order(pp, vp, m)
    loads = {mb: 1024 for mb in range(m)}
    kw = dict(num_microbatches=m, num_ranks=pp, stage0_rank=0, trainable=True, pipeline_order=order, cost_ratio=r)
    b, a = new.plan_dep(loads, **kw), old.plan_dep(loads, **kw)
    runs, step_end, times = new._idle_runs(order)
    T = times[step_end]
    bubble = sum(x.stop - x.begin for rr in runs.values() for x in rr)
    fwd_in = sum(e - s for (k, (_, s, e)) in b.placed.items() if k[0] == "encode")
    bwd_in = sum(e - s for (k, (_, s, e)) in b.placed.items() if k[0] == "backward")
    pro = max(r * len(v) for v in b.prologue.values())
    epi = max(3 * r * len(v) for v in b.epilogue.values())
    k = new.plan_dep(loads, **{**kw, "pipeline_order": None})
    k25 = (max(r * len(v) for v in k.prologue.values()) + max(3 * r * len(v) for v in k.epilogue.values())) / T
    return dict(k25=k25, r=r, bubble=bubble / (pp * T), fill=(fwd_in + bwd_in) / bubble, hidden=(fwd_in + bwd_in) / (4 * m * r),
                fwd=fwd_in / (m * r), bwd=bwd_in / (3 * m * r), nb_old=sum(1 for k in a.placed if k[0] == "backward"),
                nb=sum(1 for k in b.placed if k[0] == "backward"), exposed=(pro + epi) / T, inline=4 * m * r / T,
                stalls=len(t._stalls(pp, vp, m, b, loads)) if pp * vp * m <= 1024 else -1)

print("| layout | text : ViT | r | bubble share of the step | **fill rate** | ViT hidden (fwd / bwd) | backwards in bubbles, before -> after the fix | ViT added to the step: DEP off / K2.5 / bubble |")
print("|---|---|---:|---:|---:|---|---:|---|")
for pp, vp, m in ((2, 4, 4), (2, 4, 16), (4, 2, 16), (8, 4, 16), (8, 4, 32), (16, 2, 32), (16, 2, 64)):
    for share in (0.10, 0.25):
        x = run(pp, vp, m, share)
        flag = "" if x["stalls"] <= 0 else f" STALLS {x['stalls']}"
        print(f"| pp{pp} x vp{vp}, M{m} | {100 - round(share * 100)} : {round(share * 100)} | {x['r']:.2f} | {x['bubble']:.0%} | **{x['fill']:.0%}** | {x['hidden']:.0%} ({x['fwd']:.0%} / {x['bwd']:.0%}) | {x['nb_old']} -> {x['nb']} of {m} | +{x['inline']:.0%} / +{x['k25']:.0%} / +{x['exposed']:.0%} |{flag}")
