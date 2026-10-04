"""Draw VisionDepPlan's layout for pp 3 x vp 4, 6 micro-batches on Figure 11's timeline and compare it
box by box with the figure's transcription (fig11.json).

Text actions: torch's ScheduleInterleaved1F1B order, timed like the figure (forward periods during the
warmup ramp, then forward and backward periods alternating while forwards remain, then backwards only;
F = 1, B = 2). Vision work: the plan's prologue, epilogue and placed work, at the plan's offsets from
the edges of each idle run; the upfront column is one encode wide and the final column one backward.
"""
import json, sys
sys.path.insert(0, sys.argv[1] + '/tests/unit_tests/cpu')
from test_kimi_k3_vision_dep_plan import _interleaved_order
import torchtitan.models.kimi_k3.pipeline_parallel.vision_dep.plan as P

def rhythm(order, start):
    seqs = {r: [P.anchor_of(a) for a in acts if a is not None] for r, acts in order.items()}
    n = max(st for s in seqs.values() for _, st, _ in s) + 1
    idx = {r: 0 for r in seqs}
    done, times = set(), {}
    t, alternate, last = start, False, None
    def ready(op):
        k, st, mb = op
        if k == 'F':
            return st == 0 or ('F', st - 1, mb) in done
        return (('B', st + 1, mb) if st < n - 1 else ('F', st, mb)) in done
    while any(idx[r] < len(seqs[r]) for r in seqs):
        nxt = {r: seqs[r][idx[r]] for r in seqs if idx[r] < len(seqs[r])}
        f_left = any(op[0] == 'F' for r in seqs for op in seqs[r][idx[r]:])
        if not alternate:
            kind = 'B' if any(op[0] == 'B' and ready(op) for op in nxt.values()) else 'F'
            alternate = kind == 'B'
        else:
            kind = ('F' if last == 'B' else 'B') if f_left else 'B'
        runs = [(r, op) for r, op in nxt.items() if op[0] == kind and ready(op)]
        for r, op in runs:
            times[(r,) + op] = (t, t + P._ACTION_COST[kind]); idx[r] += 1
        done.update(op for _, op in runs)
        t += P._ACTION_COST[kind] if runs else 0.0
        last = kind
    return times, t

def layout(ratio):
    order = _interleaved_order(3, 4, 6)
    plan = P.VisionDepPlan({mb: 100 for mb in range(6)}, num_microbatches=6, num_ranks=3, stage0_rank=0,
                           trainable=True, pipeline_order=order, cost_ratio=ratio)
    runs, step_end, grid = P._idle_runs(order)
    enc, back = ratio, P._BACKWARD_COST * ratio
    text, end = rhythm(order, enc)
    boxes = {r: [] for r in range(3)}
    for (r, k, st, mb), (s, e) in text.items():
        boxes[r].append((k, st, mb + 1, s, e))
    for r in range(3):
        for i, mb in enumerate(plan.prologue[r]):
            boxes[r].append(('VF', -1, mb + 1, i * enc, (i + 1) * enc))
        for i, mb in enumerate(plan.epilogue[r]):
            boxes[r].append(('VB', -1, mb + 1, end + i * back, end + (i + 1) * back))
    for (kind, mb), (r, s, e) in plan.placed.items():
        if kind == 'encode':
            boxes[r].append(('VF', -1, mb + 1, enc + s, enc + e))
        else:
            boxes[r].append(('VB', -1, mb + 1, end - (grid[step_end] - s), end - (grid[step_end] - e)))
    return {r: sorted(b, key=lambda x: x[3]) for r, b in boxes.items()}

if __name__ == '__main__':
    fig = {int(k): [tuple(x) for x in v] for k, v in json.load(open('fig11.json')).items()}
    P._BACKWARD_COST = 2.0
    ours = layout(0.5)
    json.dump(ours, open('ours_fig_proportions.json', 'w'))
    diffs = []
    for r in range(3):
        a = {(k, st, mb): (round(s, 6), round(e, 6)) for k, st, mb, s, e in ours[r]}
        b = {(k, st, mb): (s, e) for k, st, mb, s, e in fig[r]}
        for key in sorted(set(a) | set(b), key=str):
            if a.get(key) != b.get(key):
                diffs.append((r, key, a.get(key), b.get(key)))
    n = sum(len(v) for v in fig.values())
    print(f'figure proportions (ViT forward 0.5, ViT backward 1.0): {n - len(diffs)} of {n} boxes identical')
    for d in diffs: print('  ', d)
