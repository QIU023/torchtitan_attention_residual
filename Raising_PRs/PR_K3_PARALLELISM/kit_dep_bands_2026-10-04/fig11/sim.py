import json
fig = {int(k): v for k, v in json.load(open('fig11.json')).items()}
PP, NS = 3, 12
COST = {'F': 1.0, 'B': 2.0}
def text_ops(r):
    return [(k, st, mb) for k, st, mb, s, e in fig[r] if k in ('F', 'B')]
def fig_times(r):
    return {(k, st, mb): (s, e) for k, st, mb, s, e in fig[r] if k in ('F', 'B')}

def asap(start=0.5):
    done = {}
    free = {r: start for r in range(PP)}
    seqs = {r: text_ops(r) for r in range(PP)}
    idx = {r: 0 for r in range(PP)}
    times = {}
    progress = True
    while progress:
        progress = False
        for r in range(PP):
            while idx[r] < len(seqs[r]):
                k, st, mb = seqs[r][idx[r]]
                dep = None
                if k == 'F' and st > 0: dep = ('F', st - 1, mb)
                if k == 'B' and st < NS - 1: dep = ('B', st + 1, mb)
                if dep is not None and dep not in done: break
                t = max(free[r], done.get(dep, 0.0))
                done[(k, st, mb)] = t + COST[k]
                times[(r, k, st, mb)] = (t, t + COST[k])
                free[r] = t + COST[k]
                idx[r] += 1
                progress = True
    return times

def compare(times, label):
    diffs = []
    for r in range(PP):
        ft = fig_times(r)
        for key, (s, e) in ft.items():
            if times[(r,) + key] != (s, e):
                diffs.append((r, key, times[(r,) + key], (s, e)))
    print(f'{label}: {len(diffs)} of {sum(len(fig_times(r)) for r in range(PP))} text boxes differ from the figure')
    for d in diffs[:12]: print('  ', d)
    return diffs

if __name__ == '__main__':
    compare(asap(), 'ASAP (dependencies only)')
