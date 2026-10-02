"""Old plan_dep (a93cd48ea dep_plan.py) vs new VisionDepPlan (vision_dep/plan.py): every
output field equal with the same insertion order, and the same error on bad input.

usage: python equiv_plan.py <worktree>
"""

import importlib.util
import random
import sys
from types import SimpleNamespace

HERE = __file__.rsplit("/", 1)[0].rsplit("\\", 1)[0]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


old = load("old_dep_plan", f"{HERE}/old_dep_plan.py")
new = load(
    "new_plan",
    f"{sys.argv[1]}/torchtitan/models/kimi_k3/pipeline_parallel/vision_dep/plan.py",
)

FIELDS = ("encode_rank", "backward_rank", "prologue", "epilogue", "anchored", "posts", "placed")


def canon(x):
    if isinstance(x, dict):
        return [(canon(k), canon(v)) for k, v in x.items()]
    if isinstance(x, (list, tuple)):
        return type(x)(canon(v) for v in x)
    return x


def run(fn, *a, **k):
    try:
        plan = fn(*a, **k)
    except Exception as e:  # noqa: BLE001
        return ("error", type(e).__name__, str(e))
    return tuple((f, canon(getattr(plan, f))) for f in FIELDS)


def interleaved(pp, vp, m):
    from torch.distributed.pipelining.schedules import ScheduleInterleaved1F1B

    num_stages = pp * vp
    s = ScheduleInterleaved1F1B.__new__(ScheduleInterleaved1F1B)
    s._num_stages = num_stages
    s.pp_group_size = pp
    s._n_microbatches = m
    s.n_microbatches = m
    s.stage_index_to_group_rank = {i: i % pp for i in range(num_stages)}
    s.number_of_rounds = max(1, m // pp)
    s.microbatches_per_round = m // s.number_of_rounds

    class _Stage:
        def __init__(self, i):
            self.stage_index = i
            self.num_stages = num_stages
            self.group_rank = i % pp
            self.is_first = i == 0
            self.is_last = i == num_stages - 1

    orders = {}
    for rank in range(pp):
        s._stages = [_Stage(i) for i in range(rank, num_stages, pp)]
        s.n_local_stages = len(s._stages)
        s.rank = rank
        orders[rank] = s._calculate_single_rank_operations(rank)
    return orders


KIND = ["FORWARD", "FULL_BACKWARD", "BACKWARD_INPUT", "BACKWARD_WEIGHT"]


def act(kind, stage, mb):
    return SimpleNamespace(
        computation_type=SimpleNamespace(name=kind), stage_index=stage, microbatch_index=mb
    )


def random_order(rng, R, M, stage0_rank):
    order = {}
    for r in range(R):
        L = rng.randint(0, 30)
        acts = []
        for _ in range(L):
            if rng.random() < 0.35:
                acts.append(None)
                continue
            stage = rng.randint(0, 5)
            if stage == 0 and r != stage0_rank:
                stage = 1
            acts.append(act(rng.choice(KIND), stage, rng.randrange(M)))
        order[r] = acts
    if rng.random() < 0.8:
        acts = order[stage0_rank]
        for mb in range(M):
            acts.insert(rng.randint(0, len(acts)), act("FORWARD", 0, mb))
        for mb in range(M):
            kind = rng.choice(["FULL_BACKWARD", "BACKWARD_INPUT"])
            acts.insert(rng.randint(0, len(acts)), act(kind, 0, mb))
    return order


def random_loads(rng, M):
    if rng.random() < 0.1:
        return {}
    return {
        mb: rng.choice([1, 4, 64, 256, 1024, rng.randint(1, 5000)])
        for mb in range(M)
        if rng.random() < 0.7
    }


RATIOS = (1e-3, 0.02, 0.1, 0.25, 0.32, 0.5, 1.0, 2.0, 3.61, 100.0, 0.0, -1.0)


def main():
    rng = random.Random(0)
    n = mismatch = errors = 0
    cases = []
    for pp in (1, 2, 3, 4, 8):
        for vp in (1, 2, 4):
            for m in sorted({pp, 2 * pp, 4 * pp, 8, 16, 32}):
                if m % pp:
                    continue
                order = interleaved(pp, vp, m)
                for _ in range(4):
                    loads = random_loads(rng, m) if _ else {mb: 100 for mb in range(m)}
                    for ratio in RATIOS:
                        for trainable in (True, False):
                            for bubble in (True, False):
                                cases.append(
                                    (loads, dict(num_microbatches=m, num_ranks=pp, stage0_rank=0,
                                                 trainable=trainable,
                                                 pipeline_order=order if bubble else None,
                                                 cost_ratio=ratio))
                                )
    for _ in range(20000):
        R = rng.randint(1, 6)
        M = rng.randint(1, 12)
        stage0_rank = rng.randrange(R)
        order = random_order(rng, R, M, stage0_rank)
        if rng.random() < 0.05:
            order.pop(rng.randrange(R), None)
        kwargs = dict(
            num_microbatches=M,
            num_ranks=R,
            stage0_rank=stage0_rank,
            trainable=rng.random() < 0.7,
            pipeline_order=order if rng.random() < 0.85 else None,
            cost_ratio=rng.choice(RATIOS + (rng.uniform(0.001, 10.0),)),
        )
        cases.append((random_loads(rng, M), kwargs))
    placed_total = 0
    for loads, kwargs in cases:
        a = run(old.plan_dep, dict(loads), **kwargs)
        b = run(new.VisionDepPlan, dict(loads), **kwargs)
        n += 1
        if a[0] == "error":
            errors += 1
        else:
            placed_total += len(dict(a)["placed"])
        if a != b:
            mismatch += 1
            if mismatch <= 3:
                print("MISMATCH", kwargs.get("num_ranks"), kwargs.get("cost_ratio"))
                print(" old:", a if a[0] == "error" else [f for f, v in a if v != dict(b).get(f)])
                print(" new:", b if b[0] == "error" else "")
    print(f"cases {n}, errors (same on both) {errors}, placed work items {placed_total}, mismatches {mismatch}")
    sys.exit(1 if mismatch else 0)


main()
