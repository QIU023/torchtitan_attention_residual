import sys
sys.path.insert(0, '/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_dep_1004/tests/unit_tests/cpu')
from test_kimi_k3_vision_dep_plan import _interleaved_order
import torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan as P
P._BACKWARD_COST = 2.0  # the figure draws a ViT backward at twice its forward
order = _interleaved_order(3, 4, 6)
runs, step_end, times = P._idle_runs(order)
T = times[step_end]
for ratio in (0.25, 0.5, 0.51):
    plan = P.VisionDepPlan({m: 100 for m in range(6)}, num_microbatches=6, num_ranks=3, stage0_rank=0,
                           trainable=True, pipeline_order=order, cost_ratio=ratio)
    enc = {m + 1: plan.placed[('encode', m)] for m in range(6) if ('encode', m) in plan.placed}
    bwd = {m + 1: (r, s - T, e - T) for (k, m), (r, s, e) in plan.placed.items() if k == 'backward'}
    print(ratio, {r: tuple(m + 1 for m in v) for r, v in plan.prologue.items()}, {r: tuple(m + 1 for m in v) for r, v in plan.epilogue.items()})
    print('   encodes', enc)
    print('   backwards', bwd)
