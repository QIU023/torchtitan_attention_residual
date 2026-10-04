import re, html, json
from scan import rows, t0, u
s = open('p19_bbox.html').read()
words = re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">([^<]*)</word>', s)
labels = {121: [], 137: [], 153: []}
for x0, y0, x1, y1, w in words:
    x0, y0, x1 = float(x0), float(y0), float(x1)
    w = html.unescape(w)
    if 97 <= x0 <= 545 and round(y0) in labels and w.isdigit():
        n = len(w)
        for i, ch in enumerate(w):
            cx = x0 + (x1 - x0) * (i + 0.5) / n
            labels[round(y0)].append(((cx - t0) / u, int(ch)))
out = {}
for r, (name, y) in enumerate((('PP0', 121), ('PP1', 137), ('PP2', 153))):
    ops = []
    for s_, e, k in rows[name]:
        if k in ('.', 'DL'):
            continue
        mbs = [mb for x, mb in labels[y] if s_ <= x < e]
        assert len(mbs) == 1, (name, s_, e, k, mbs)
        mb = mbs[0]
        if k in ('VF', 'VB'):
            ops.append((k, -1, mb, s_, e))
        else:
            chunk = int(k[1])
            ops.append((k[0], chunk * 3 + r, mb, s_, e))
    out[r] = ops
json.dump(out, open('fig11.json', 'w'))
for r, ops in out.items():
    print(f'PP{r}:', ' '.join(f'{k}{st if st >= 0 else ""}.{mb}@{s:g}' for k, st, mb, s, e in ops))
