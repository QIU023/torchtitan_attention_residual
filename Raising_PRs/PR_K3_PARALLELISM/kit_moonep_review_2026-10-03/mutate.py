import sys
path, kind = sys.argv[1], sys.argv[2]
s = open(path).read()
if kind == "no_refill":
    old = "        for name, row in prefetch_rows(buffer, plan, group, w13, w2).items()\n"
    new = "        for name, row in _stale_rows(group, w13, w2).items()\n"
    helper = '''

def _stale_rows(group, w13, w2):
    from torchtitan.distributed.moonep.moonep import _pools_for, _projections, _PROJECTIONS
    local = _projections(w13, w2)
    shapes = {n: tuple(w.shape[1:]) for n, w in local.items()}
    pools = _pools_for(group, w2.shape[0], shapes, torch.bfloat16, "weight")
    rank = torch.distributed.get_rank(group)
    return {n: pools[n][rank] for n in _PROJECTIONS}
'''
    assert s.count(old) == 1
    s = s.replace(old, new) + helper
elif kind == "last_plan":
    old_d = "    _plans[_next_plan_id] = _Plan(plan)\n"
    new_d = "    _plans[_next_plan_id] = _Plan(plan)\n    _LAST[0] = plan\n"
    old_c = "        grad_out.to(torch.bfloat16).contiguous(), plan=ctx.entry.plan\n"
    new_c = "        grad_out.to(torch.bfloat16).contiguous(), plan=_LAST[0]\n"
    assert s.count(old_d) == 1 and s.count(old_c) == 1
    s = s.replace(old_d, new_d).replace(old_c, new_c).replace("_next_plan_id = 0\n", "_next_plan_id = 0\n_LAST = [None]\n", 1)
open(path, "w").write(s)
print("mutated", kind)
