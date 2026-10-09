"""#4381 DEP planner (095acaaad) against the closed form on interleaved 1F1B (v >= 2), every micro-batch one equal image;
r values chosen so no placement lands exactly on a run's end (float accumulation decides those). Logbook kit only."""
import itertools
import math

from tests.unit_tests.cpu.test_kimi_k3_vision_dep_plan import _interleaved_order
from torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan import VisionDepPlan


def closed_form(M, p, r):
    f_cap = sum(math.floor(i / r) for i in range(1, p))
    b_cap = sum(math.floor((2 * i - 1) / (3 * r)) for i in range(1, p))
    return min(M - p, f_cap), min(M - p, b_cap)


points, bad = 0, []
for p, v, k, r in itertools.product((2, 3, 4, 6, 8), (2, 3, 4), (1, 2, 3, 4, 8, 16), (0.03, 0.07, 0.13, 0.17, 0.27, 0.45, 0.9, 1.7)):
    M = k * p
    order = _interleaved_order(p, v, M)
    plan = VisionDepPlan({m: 100 for m in range(M)}, num_microbatches=M, num_ranks=p, stage0_rank=0,
                         trainable=True, pipeline_order=order, cost_ratio=r)
    got = (sum(1 for kind, _ in plan.placed if kind == "encode"), sum(1 for kind, _ in plan.placed if kind == "backward"))
    points += 1
    if got != closed_form(M, p, r):
        bad.append((p, v, M, r, got, closed_form(M, p, r)))
print(f"v>=2 grid points {points}, mismatches {len(bad)}")
for b in bad[:10]:
    print("MISMATCH", b)
# Figure 11 and the measured cell
for p, v, M, r in ((3, 4, 6, 0.1), (3, 4, 6, 0.3), (4, 4, 16, 0.046), (4, 4, 16, 0.136), (4, 4, 16, 0.169), (8, 4, 32, 0.1), (8, 4, 64, 0.1)):
    order = _interleaved_order(p, v, M)
    plan = VisionDepPlan({m: 100 for m in range(M)}, num_microbatches=M, num_ranks=p, stage0_rank=0, trainable=True, pipeline_order=order, cost_ratio=r)
    e = sum(1 for kind, _ in plan.placed if kind == "encode"); b = sum(1 for kind, _ in plan.placed if kind == "backward")
    print(f"p{p} v{v} M{M} r{r}: fwd in bubbles {e}/{M} = {e/M:.3f}, bwd {b}/{M} = {b/M:.3f}, 1-p/M = {1-p/M:.3f}, closed form {closed_form(M, p, r)}")
