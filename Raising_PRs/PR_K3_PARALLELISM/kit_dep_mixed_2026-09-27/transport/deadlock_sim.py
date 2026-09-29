"""Worst-case lazy-loading model: a rank runs no compute while any P2P op it posted is unmatched.

Checks the new DEP transport (plan_dep posts) and the old one (all receives at step start,
sends right after the data exists) against torch's own Interleaved1F1B send/recv order.
"""

import importlib.util
import random
import sys

from torch.distributed.pipelining import schedules as S

spec = importlib.util.spec_from_file_location(
    "dep_plan",
    "C:/Users/78532/AppData/Local/Temp/claude/dep/torchtitan/models/kimi_k3/pipeline_parallel/dep_plan.py",
)
dp = importlib.util.module_from_spec(spec)
sys.modules["dep_plan"] = dp
spec.loader.exec_module(dp)
from order_probe import order  # noqa: E402


def comms_order(pp, vp, m):
    compute = order(pp, vp, m)
    num_stages = pp * vp
    trimmed = {}
    for r, acts in compute.items():
        last = max(i for i, a in enumerate(acts) if a is not None)
        trimmed[r] = list(acts[: last + 1])
    return compute, S._add_send_recv(
        {r: list(v) for r, v in trimmed.items()}, lambda s: s % pp, num_stages
    )


def kind_of(a):
    return getattr(a.computation_type, "name", str(a.computation_type))


def build_events(pp, vp, m, plan, mode, loads):
    """Per rank, the ordered events the runtime issues."""
    compute, comms = comms_order(pp, vp, m)
    num_stages = pp * vp
    events = {r: [] for r in range(pp)}
    for r in range(pp):
        ev = events[r]
        if mode == "none":
            pass
        elif mode == "old":
            if r == 0:
                ev += [("post", ("recv_feature", mb, plan.encode_rank[mb])) for mb in sorted(loads) if plan.encode_rank[mb] != 0]
            else:
                ev += [("post", ("recv_gradient", mb, 0)) for mb in sorted(loads) if plan.backward_rank.get(mb) == r]
            for mb in plan.prologue[r]:
                ev.append(("compute", ("encode", mb)))
                if r != 0:
                    ev.append(("post", ("send_feature", mb, 0)))
            for item in plan.anchored[r].get(dp.START, ()):
                ev.append(("compute", item))
                if item[0] == "encode" and r != 0:
                    ev.append(("post", ("send_feature", item[1], 0)))
        else:
            ev += [("compute", ("encode", mb)) for mb in plan.prologue[r]]
            ev += [("post", t) for t in plan.posts[r].get(dp.STEP_START, ())]
            ev += [("compute", item) for item in plan.anchored[r].get(dp.START, ())]
        acts = comms[r]
        i = 0
        while i < len(acts):
            a = acts[i]
            k = kind_of(a)
            if k in ("FORWARD", "FULL_BACKWARD", "BACKWARD_INPUT", "BACKWARD_WEIGHT"):
                anchor = dp.anchor_of(a)
                if mode == "new":
                    ev += [("post", t) for t in plan.posts[r].get(("before", anchor), ())]
                if k == "FORWARD" and a.stage_index == 0 and mode != "none" and a.microbatch_index in loads:
                    ev.append(("use_feature", a.microbatch_index))
                ev.append(("action", a))
                after = []
                if mode != "none":
                    if mode == "new":
                        after += [("post", t) for t in plan.posts[r].get(("after", anchor), ())]
                    for item in plan.anchored[r].get(anchor, ()):
                        after.append(("compute", item))
                        if mode == "old" and item[0] == "encode" and r != 0:
                            after.append(("post", ("send_feature", item[1], 0)))
                    if mode == "old" and k in ("FULL_BACKWARD", "BACKWARD_INPUT") and a.stage_index == 0 and a.microbatch_index in plan.backward_rank:
                        e = plan.backward_rank[a.microbatch_index]
                        if e != 0:
                            after.insert(0, ("post", ("send_gradient", a.microbatch_index, e)))
                if i + 1 < len(acts) and kind_of(acts[i + 1]) in ("SEND_F", "SEND_B") and acts[i + 1].stage_index == a.stage_index and acts[i + 1].microbatch_index == a.microbatch_index:
                    ev.append(("sched", acts[i + 1]))
                    i += 1
                ev += after
            else:
                ev.append(("sched", a))
            i += 1
        if mode == "new":
            ev += [("post", t) for t in plan.posts[r].get(dp.STEP_END, ())]
        if mode in ("new", "old"):
            ev += [("compute", ("backward", mb)) for mb in plan.epilogue[r]]
    return events, num_stages


def simulate(pp, vp, m, plan, mode, loads):
    events, num_stages = build_events(pp, vp, m, plan, mode, loads)
    ptr = {r: 0 for r in range(pp)}
    posted = set()  # (rank, key)
    done = set()
    pending = {r: set() for r in range(pp)}

    def sched_key(a):
        k = kind_of(a)
        if k == "SEND_F":
            return ("F", a.stage_index, a.microbatch_index), "send"
        if k == "RECV_F":
            return ("F", a.stage_index - 1, a.microbatch_index), "recv"
        if k == "SEND_B":
            return ("B", a.stage_index, a.microbatch_index), "send"
        return ("B", a.stage_index + 1, a.microbatch_index), "recv"

    def partner_posted(key):
        return key in posted

    def refresh():
        for r in range(pp):
            pending[r] = {p for p in pending[r] if p[1] not in posted}

    progress = True
    while progress:
        progress = False
        for r in range(pp):
            while ptr[r] < len(events[r]):
                kind, obj = events[r][ptr[r]]
                if kind == "post":
                    t_kind, mb, peer = obj
                    direction, what = t_kind.split("_")
                    me = ("dep", what, mb, direction)
                    partner = ("dep", what, mb, "recv" if direction == "send" else "send")
                    posted.add(me)
                    pending[r].add((me, partner))
                elif kind == "sched":
                    key, direction = sched_key(obj)
                    me = ("sched", key, direction)
                    partner = ("sched", key, "recv" if direction == "send" else "send")
                    posted.add(me)
                    pending[r].add((me, partner))
                else:
                    refresh()
                    if pending[r]:
                        break  # worst case: every kernel waits for the rank's unmatched ops
                    if kind == "use_feature":
                        if ("dep", "feature", obj, "send") not in posted and ("encode", obj) not in done:
                            break
                    elif kind == "action":
                        k = kind_of(obj)
                        s, mb = obj.stage_index, obj.microbatch_index
                        if k == "FORWARD" and s > 0 and ("F", s - 1, mb) not in done:
                            break
                        if k in ("FULL_BACKWARD", "BACKWARD_INPUT"):
                            if s < num_stages - 1 and ("B", s + 1, mb) not in done:
                                break
                        done.add(("F" if k == "FORWARD" else "B", s, mb))
                    elif kind == "compute":
                        what, mb = obj
                        if what == "backward":
                            if plan.backward_rank[mb] != 0 and ("dep", "gradient", mb, "send") not in posted:
                                break
                            if ("B", 0, mb) not in done:
                                break
                        done.add(obj)
                ptr[r] += 1
                progress = True
            refresh()
    stuck = {r: events[r][ptr[r]] for r in range(pp) if ptr[r] < len(events[r])}
    return stuck


def uneven(m):
    patches = (64, 0, 1024, 256, 64, 256, 0, 1024)
    return {mb: patches[mb % 8] for mb in range(m) if patches[mb % 8]}


results = []
for pp, vp, m in ((2, 4, 4), (4, 2, 8), (4, 2, 16), (8, 4, 16), (8, 4, 32)):
    for loads in ({mb: 100 for mb in range(m)}, uneven(m)):
        for bubble in (False, True):
            for ratio in (0.25, 1.0):
                compute, _ = comms_order(pp, vp, m)
                plan = dp.plan_dep(
                    loads,
                    num_microbatches=m,
                    num_ranks=pp,
                    stage0_rank=0,
                    trainable=True,
                    pipeline_order=compute if bubble else None,
                    cost_ratio=ratio,
                )
                base = simulate(pp, vp, m, plan, "none", loads)
                new = simulate(pp, vp, m, plan, "new", loads)
                old = simulate(pp, vp, m, plan, "old", loads)
                results.append((pp, vp, m, len(loads), bubble, ratio, not base, not new, not old))
for r in results:
    print("pp%d vp%d M%d imgs%d bubble=%s ratio=%.2f | torch alone ok=%s | new ok=%s | old ok=%s" % r)
print("all torch-alone ok:", all(r[6] for r in results))
print("all new ok:", all(r[7] for r in results))
print("old deadlocks:", sum(not r[8] for r in results), "of", len(results))
