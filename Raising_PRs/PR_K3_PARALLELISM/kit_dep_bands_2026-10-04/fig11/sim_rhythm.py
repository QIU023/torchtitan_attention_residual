"""Lockstep 1F1B rhythm: forward periods during the warmup ramp; from the first ready backward on,
forward and backward periods alternate while any forward remains; then backward periods only.
In each period every rank runs its next action if it has the period's kind and its input was
produced in an earlier period."""
from sim import text_ops, compare, PP, NS, COST

def simulate(start=0.5):
    seqs = {r: text_ops(r) for r in range(PP)}
    idx = {r: 0 for r in range(PP)}
    done_period = {}
    times = {}
    t, period, alternate, last_kind = start, 0, False, None
    def ready(op):
        k, st, mb = op
        dep = None
        if k == 'F' and st > 0: dep = ('F', st - 1, mb)
        if k == 'B': dep = ('B', st + 1, mb) if st < NS - 1 else ('F', st, mb)
        return dep is None or dep in done_period
    while any(idx[r] < len(seqs[r]) for r in range(PP)):
        nxt = {r: seqs[r][idx[r]] for r in range(PP) if idx[r] < len(seqs[r])}
        b_ready = any(op[0] == 'B' and ready(op) for op in nxt.values())
        f_left = any(op[0] == 'F' for r in range(PP) for op in seqs[r][idx[r]:])
        if not alternate:
            kind = 'B' if b_ready else 'F'
            if kind == 'B': alternate = True
        else:
            kind = ('F' if last_kind == 'B' else 'B') if f_left else 'B'
        runs = [(r, op) for r, op in nxt.items() if op[0] == kind and ready(op)]
        for r, op in runs:
            times[(r,) + op] = (t, t + COST[kind])
            idx[r] += 1
        for r, op in runs:
            done_period[op] = period
        t += COST[kind] if runs else 0.0
        last_kind = kind
        period += 1
        if period > 1000: raise RuntimeError('stuck')
    return times

if __name__ == '__main__':
    compare(simulate(), 'lockstep 1F1B rhythm')
