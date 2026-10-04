"""Variant: matched exchanges complete together (a call waits for every call it is matched with to complete)."""
import sim_megatron as m
from sim import compare, COST

def simulate(start=0.5):
    progs = m.programs()
    where_send, where_recv = {}, {}
    for r, steps in progs.items():
        for i, s in enumerate(steps):
            for t in s['sends']: where_send[t] = (r, i)
            for t in s['recvs']: where_recv[t] = (r, i)
    post = {(r, i): 0.0 for r in progs for i in range(len(progs[r]))}
    done = dict(post)
    for _ in range(1000):
        changed = False
        for r, steps in progs.items():
            prev = start
            for i, s in enumerate(steps):
                p = prev + sum(COST[op[0]] for op in s['compute'])
                peers = [where_recv[t] for t in s['sends']] + [where_send[t] for t in s['recvs']]
                d = max([p] + [post[pe] for pe in peers] + [done[pe] for pe in peers])
                if p != post[(r, i)] or d != done[(r, i)]:
                    changed = True
                post[(r, i)], done[(r, i)] = p, d
                prev = d
        if not changed: break
    times = {}
    for r, steps in progs.items():
        t = start
        for i, s in enumerate(steps):
            for op in s['compute']:
                times[(r,) + op] = (t, t + COST[op[0]]); t += COST[op[0]]
            t = done[(r, i)]
    return times

if __name__ == '__main__':
    compare(simulate(), 'matched exchanges complete together')
