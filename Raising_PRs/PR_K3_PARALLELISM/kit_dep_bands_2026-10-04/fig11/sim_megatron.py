"""Megatron-style interleaved 1F1B timing: every step ends with one blocking batched exchange,
which completes when each peer has posted the matching op."""
from sim import fig, text_ops, compare, PP, NS, COST

def programs():
    progs = {}
    for r in range(PP):
        seq = text_ops(r)
        fwd = [op for op in seq if op[0] == 'F']
        bwd = [op for op in seq if op[0] == 'B']
        w = {0: 13, 1: 11, 2: 9}[r]
        steps = []
        def recv_for(op):
            k, st, mb = op
            if k == 'F' and st > 0: return [('act', st - 1, mb)]
            if k == 'B' and st < NS - 1: return [('grad', st + 1, mb)]
            return []
        def send_of(op):
            k, st, mb = op
            if k == 'F' and st < NS - 1: return [('act', st, mb)]
            if k == 'B' and st > 0: return [('grad', st, mb)]
            return []
        steps.append({'compute': [], 'sends': [], 'recvs': recv_for(fwd[0])})
        n_fb = len(fwd) - w
        for k in range(w):
            recvs = recv_for(fwd[k + 1]) if k + 1 < len(fwd) else []
            if k == w - 1 and n_fb >= 0:
                recvs = recvs + recv_for(bwd[0])
            steps.append({'compute': [fwd[k]], 'sends': send_of(fwd[k]), 'recvs': recvs})
        for k in range(n_fb):
            f, b = fwd[w + k], bwd[k]
            recvs = []
            if w + k + 1 < len(fwd): recvs += recv_for(fwd[w + k + 1])
            if k + 1 < len(bwd): recvs += recv_for(bwd[k + 1])
            steps.append({'compute': [f, b], 'sends': send_of(f) + send_of(b), 'recvs': recvs})
        for k in range(n_fb, len(bwd)):
            b = bwd[k]
            recvs = recv_for(bwd[k + 1]) if k + 1 < len(bwd) else []
            steps.append({'compute': [b], 'sends': send_of(b), 'recvs': recvs})
        progs[r] = steps
    return progs

def simulate(start=0.5):
    progs = programs()
    where_send, where_recv = {}, {}
    for r, steps in progs.items():
        for i, s in enumerate(steps):
            for t in s['sends']: where_send[t] = (r, i)
            for t in s['recvs']: where_recv[t] = (r, i)
    post = {(r, i): 0.0 for r in progs for i in range(len(progs[r]))}
    done = dict(post)
    for _ in range(500):
        changed = False
        for r, steps in progs.items():
            prev = start
            for i, s in enumerate(steps):
                p = prev + sum(COST[op[0]] for op in s['compute'])
                peers = [where_recv[t] for t in s['sends']] + [where_send[t] for t in s['recvs']]
                d = max([p] + [post[pe] for pe in peers])
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
    compare(simulate(), 'Megatron-style blocking exchanges')
