import sys
sys.path.insert(0, '/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_dep_1004/tests/unit_tests/cpu')
from test_kimi_k3_vision_dep_plan import _interleaved_order
from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan import VisionDepPlan, _idle_runs

order = _interleaved_order(3, 4, 6)
runs, step_end, times = _idle_runs(order)
T = times[step_end]
for ratio in (0.01, 0.05, 0.1, 0.2, 0.3, 1/3, 0.34, 0.5, 1.0):
    plan = VisionDepPlan({m: 100 for m in range(6)}, num_microbatches=6, num_ranks=3, stage0_rank=0,
                         trainable=True, pipeline_order=order, cost_ratio=ratio)
    enc = {m + 1: plan.placed[('encode', m)] for m in range(6) if ('encode', m) in plan.placed}
    bwd = {m + 1: (r, round(s - T, 3), round(e - T, 3)) for (k, m), (r, s, e) in plan.placed.items() if k == 'backward'}
    pro = {r: tuple(m + 1 for m in v) for r, v in plan.prologue.items()}
    epi = {r: tuple(m + 1 for m in v) for r, v in plan.epilogue.items()}
    fig = (pro == {0: (1,), 1: (2,), 2: (3,)} and epi == {0: (4,), 1: (5,), 2: (6,)}
           and {m: v[0] for m, v in enc.items()} == {4: 1, 5: 2, 6: 2}
           and {m: v[0] for m, v in bwd.items()} == {1: 1, 2: 2, 3: 2})
    print(f'ratio {ratio:.3f}: figure layout {fig}; prologue {pro} epilogue {epi}')
    print(f'   encodes {enc}')
    print(f'   backwards (times relative to the step end {T}) {bwd}')
