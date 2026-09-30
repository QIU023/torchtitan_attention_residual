import importlib.util, sys
W = sys.argv[1]
sys.path.insert(0, W)
spec = importlib.util.spec_from_file_location("t", f"{W}/tests/unit_tests/cpu/test_kimi_k3_dep_plan.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
from torchtitan.models.kimi_k3.pipeline_parallel.dep_plan import plan_dep, _idle_runs
for pp, vp, m in ((2, 4, 4), (2, 4, 16), (4, 2, 16), (8, 4, 16), (8, 4, 32)):
    order = t._interleaved_order(pp, vp, m)
    loads = {mb: 1024 for mb in range(m)}
    for ratio in (0.5, 1.0):
        plan = plan_dep(loads, num_microbatches=m, num_ranks=pp, stage0_rank=0, trainable=True,
                        pipeline_order=order, cost_ratio=ratio)
        enc_idle = sum(1 for k in plan.placed if k[0] == "encode")
        bwd_idle = sum(1 for k in plan.placed if k[0] == "backward")
        pro = sum(len(v) for v in plan.prologue.values()); epi = sum(len(v) for v in plan.epilogue.values())
        stall = t._stalls(pp, vp, m, plan, loads)
        print(f"pp{pp} x vp{vp} M{m} ratio {ratio}: encodes idle {enc_idle}/{m} (prologue {pro}), backwards idle {bwd_idle}/{m} (epilogue {epi}), stalls {len(stall)}")
