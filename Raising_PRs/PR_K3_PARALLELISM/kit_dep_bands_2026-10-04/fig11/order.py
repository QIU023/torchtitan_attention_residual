import json, sys
sys.path.insert(0, '/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/wt_dep_1004/tests/unit_tests/cpu')
from test_kimi_k3_vision_dep_plan import _interleaved_order
fig = {int(k): v for k, v in json.load(open('fig11.json')).items()}
order = _interleaved_order(3, 4, 6)
ok = True
for r in range(3):
    torch_seq = [({'FORWARD': 'F', 'FULL_BACKWARD': 'B'}[a.computation_type.name], int(a.stage_index), int(a.microbatch_index) + 1) for a in order[r] if a is not None]
    fig_seq = [(k, st, mb) for k, st, mb, s, e in fig[r] if k in ('F', 'B')]
    same = torch_seq == fig_seq
    ok &= same
    print(f'PP{r}: {len(torch_seq)} torch actions, {len(fig_seq)} figure actions, same order: {same}')
    if not same:
        for i, (a, b) in enumerate(zip(torch_seq, fig_seq)):
            if a != b:
                print('  first difference at', i, a, b); break
print('ALL SAME' if ok else 'DIFFER')
