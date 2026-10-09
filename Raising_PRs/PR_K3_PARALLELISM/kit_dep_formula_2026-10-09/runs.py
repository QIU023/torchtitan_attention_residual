import sys
from tests.unit_tests.cpu.test_kimi_k3_vision_dep_plan import _interleaved_order
from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan import _idle_runs, anchor_of
for p, v, M in ((2, 1, 8), (3, 1, 6), (3, 2, 6), (3, 4, 6), (4, 2, 16), (4, 4, 16)):
    order = _interleaved_order(p, v, M)
    runs, step_end, times = _idle_runs(order)
    opening = {r: (rs[0].begin, rs[0].stop) for r, rs in runs.items() if rs and rs[0].start == 0}
    closing = {r: (rs[-1].begin, rs[-1].stop) for r, rs in runs.items() if rs and rs[-1].end == step_end}
    mid = {r: [(round(x.begin, 1), round(x.stop, 1)) for x in rs if x.start != 0 and x.end != step_end] for r, rs in runs.items()}
    print(f"p{p} v{v} M{M}: step {times[-1]:.0f} | opening {opening} | closing {closing} | mid-idle {mid}")
    if v == 1 and p == 2:
        for r, acts in order.items():
            print("   rank", r, " ".join("." if a is None else "".join(map(str, anchor_of(a)[:1])) + str(anchor_of(a)[2]) for a in acts))
