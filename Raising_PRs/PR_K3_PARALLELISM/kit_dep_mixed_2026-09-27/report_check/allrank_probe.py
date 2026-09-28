from order_probe import order
from dep_bubble_plan import plan_for_rank

def kind(a):
    t = str(a.computation_type)
    return "F" if "FORWARD" in t else ("B" if "BACKWARD" in t else "?")

def all_ranks(pp, vp, m, ratio=1.0, hop=1):
    po = order(pp, vp, m)
    T = max(len(v) for v in po.values())
    idle = {r: [i for i, a in enumerate(po[r]) if a is None and i < max(j for j, b in enumerate(po[r]) if b is not None)] for r in po}
    consume = {a.microbatch_index: i for i, a in enumerate(po[0]) if a is not None and kind(a) == "F" and a.stage_index == 0}
    grad_at = {a.microbatch_index: i for i, a in enumerate(po[0]) if a is not None and kind(a) == "B" and a.stage_index == 0}
    free = {r: list(v) for r, v in idle.items()}
    budget_cost = ratio
    # forward: mb >= pp need an encode finished `hop` slots before rank 0 consumes it
    fwd_hidden = []
    for mb in sorted(consume):
        if mb < pp:
            continue
        deadline = consume[mb] - hop
        best = None
        for r, slots in free.items():
            usable = [s for s in slots if s < deadline]
            need = int(-(-budget_cost // 1))
            if len(usable) >= need:
                pick = usable[-need:]
                if best is None or pick[-1] > best[1][-1]:
                    best = (r, pick)
        if best:
            r, pick = best
            for s in pick:
                free[r].remove(s)
            fwd_hidden.append((mb, r))
    # backward: tower backward of mb after rank 0's B0.mb (+hop to reach the rank)
    bwd_hidden = []
    for mb in sorted(grad_at):
        start = grad_at[mb] + hop
        best = None
        for r, slots in free.items():
            usable = [s for s in slots if s > start]
            need = int(-(-2 * budget_cost // 1))
            if len(usable) >= need:
                pick = usable[:need]
                if best is None or pick[0] < best[1][0]:
                    best = (r, pick)
        if best:
            r, pick = best
            for s in pick:
                free[r].remove(s)
            bwd_hidden.append((mb, r))
    p0 = plan_for_rank(po[0], rank=0, vision_microbatches=m, cost_ratio=ratio, upfront=min(pp, m), vision_stage=0)
    return fwd_hidden, bwd_hidden, p0

for pp, vp, m in [(4, 2, 8), (4, 2, 16), (8, 4, 16), (8, 4, 32)]:
    f, b, p0 = all_ranks(pp, vp, m)
    print(f"pp{pp} x vp{vp} M{m}: tower on rank 0 only (branch plan): fwd in bubbles {len(p0.placed)}/{m}, upfront {len(p0.upfront)}, inline {len(p0.synchronous)} | "
          f"tower on every rank: fwd in bubbles {len(f)}/{m} (upfront {pp}), bwd in bubbles {len(b)}/{m}; fwd ranks {sorted(set(r for _, r in f))}")
