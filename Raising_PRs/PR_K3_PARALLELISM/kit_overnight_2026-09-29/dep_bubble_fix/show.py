import importlib.util, sys
W, pp, vp, m = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
sys.path.insert(0, W)
spec = importlib.util.spec_from_file_location("t", f"{W}/tests/unit_tests/cpu/test_kimi_k3_dep_plan.py")
t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
from torchtitan.models.kimi_k3.pipeline_parallel.dep_plan import _idle_runs, _stage0_slots, anchor_of
order = t._interleaved_order(pp, vp, m)
for r, acts in order.items():
    cells = []
    for a in acts:
        if a is None: cells.append("  .  ")
        else:
            k, s, mb = anchor_of(a); cells.append(f"{k}{s}m{mb}".ljust(5))
    print(f"rank {r}: " + "|".join(cells))
runs, step_end = _idle_runs(order)
print("step_end", step_end)
for r, rr in runs.items():
    print(f"rank {r} idle runs:", [(x.start, x.end) for x in rr])
consume, ready = _stage0_slots(order[0])
print("stage0 consume:", consume)
print("stage0 ready:  ", ready)
