"""The planner's own model: torch's lockstep action grid, a slot lasting its longest action."""
import sys
sys.path.insert(0, '/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_dep_1004/tests/unit_tests/cpu')
from test_kimi_k3_vision_dep_plan import _interleaved_order
from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan import _idle_runs, anchor_of
from sim import compare

def simulate(start=0.5):
    order = _interleaved_order(3, 4, 6)
    runs, step_end, times = _idle_runs(order)
    out = {}
    for r, acts in order.items():
        for slot, a in enumerate(acts):
            if a is None: continue
            k, st, mb = anchor_of(a)
            out[(r, k, st, mb + 1)] = (start + times[slot], start + times[slot + 1] if k == 'B' or times[slot+1]-times[slot] == 1 else start + times[slot] + 1)
    return out

if __name__ == '__main__':
    d = compare(simulate(), "planner's lockstep grid")
