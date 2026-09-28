import sys
from torch.distributed.pipelining import schedules as S
from dep_bubble_plan import plan_for_rank

def order(pp, vp, m):
    o = S.ScheduleInterleaved1F1B.__new__(S.ScheduleInterleaved1F1B)
    o.pp_group_size, o.n_local_stages, o._n_microbatches = pp, vp, m
    o.number_of_rounds = max(1, m // pp)
    o.microbatches_per_round = m // o.number_of_rounds
    o._num_stages = pp * vp
    return {r: o._calculate_single_rank_operations(r) for r in range(pp)}

def fmt(a):
    if a is None:
        return "."
    t = str(a.computation_type)
    k = "F" if "FORWARD" in t else ("B" if "BACKWARD" in t else "?")
    return f"{k}{a.stage_index}.{a.microbatch_index}"

if __name__ == "__main__":
    for pp, vp, m in [(4, 2, 8), (8, 4, 16)]:
        po = order(pp, vp, m)
        print(f"== pp{pp} x vp{vp}, M{m}")
        for r in range(pp):
            acts = po[r]
            last = max(i for i, a in enumerate(acts) if a is not None)
            body = acts[: last + 1]
            idle = sum(a is None for a in body)
            lead = next(i for i, a in enumerate(body) if a is not None)
            print(f"rank {r}: {len(body)} slots, idle {idle} (leading {lead})")
            if pp == 4:
                print("   ", " ".join(fmt(a) for a in body))
        for ratio in (1.0, 2.0):
            p = plan_for_rank(po[0], rank=0, vision_microbatches=m, cost_ratio=ratio, upfront=min(pp, m), vision_stage=0)
            print(f"rank 0 plan, cost_ratio {ratio}: upfront {len(p.upfront)}, placed {len(p.placed)}, synchronous {len(p.synchronous)}, idle slots {p.idle_slots}")
