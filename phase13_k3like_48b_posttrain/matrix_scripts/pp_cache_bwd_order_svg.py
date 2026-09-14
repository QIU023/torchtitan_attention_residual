"""PP rank cache, pp4 x vp4, 4 blocks x 4 layers: forward payloads and backward summation order, as one SVG.

Trees are built from the code's rule and checked against PP_CACHE_REDUCTION_ORDER_2026-09-13.md section 4b.
"""
import html
import re
import sys

OUT = sys.argv[1]

P, S = 4, 16
ENTRIES = ["e", "x4", "x8", "x12"]
PROD = {"e": 0, "x4": 3, "x8": 7, "x12": 11}
GRAD = {"e": "de", "x4": "dx4", "x8": "dx8", "x12": "dx12"}
FILL = ["#dbeafe", "#d1fae5", "#fef3c7", "#ede9fe"]
STROKE = ["#2563eb", "#059669", "#d97706", "#7c3aed"]
SANS = "'Segoe UI', Inter, Helvetica, Arial, sans-serif"
MONO = "Consolas, 'JetBrains Mono', Menlo, monospace"
INK, MUTED = "#111827", "#6b7280"
W = 1480


def rank(s):
    return s % P


def layers(s):
    if s == 0:
        return "emb, layer 0"
    if s == 1:
        return "layers 1-2"
    if s == S - 1:
        return "head"
    return f"layer {s + 1}"


def naive_payload(s):
    return [y for y in ENTRIES if PROD[y] <= s]


def cache_payload(s):
    return [y for y in ENTRIES if s + 1 - P < PROD[y] <= s]


def store_at_entry(s):
    return [y for y in ENTRIES if PROD[y] <= s - P]


def commits(s):
    return [y for y in ENTRIES if PROD[y] == s]


# the r3998924467 tables
assert cache_payload(3) == ["x4"] and cache_payload(6) == [] and cache_payload(14) == []
assert cache_payload(7) == ["x8"] and cache_payload(13) == ["x12"]
assert naive_payload(14) == ENTRIES and store_at_entry(11) == ["e", "x4", "x8"]
assert sum(len(naive_payload(s)) for s in range(S - 1)) == 39
assert sum(len(cache_payload(s)) for s in range(S - 1)) == 12


def naive_items(y):
    p = PROD[y]
    return [("leaf", s, None if s == S - 1 else s) for s in range(S - 1, p - 1, -1)]


def readers(h):
    """Later stages on h's rank, in deposit order (highest stage first)."""
    return list(range(h + P, S, P))[::-1]


def cache_items(y):
    p = PROD[y]
    holders = list(range(p, min(p + P, S)))
    items = []
    for h in reversed(holders):
        dep = readers(h)
        if h != p:  # received: own backward first, then the rank's deposit (backward_one_chunk)
            items.append(("leaf", h, h if items else None))
            if dep:
                items.append(("group", dep, h))
        else:  # committed: the deposit first (_retrieve_recv_grads), then own backward
            if dep:
                items.append(("group", dep, h))
            items.append(("leaf", h, h))
    return items


def canon_items(items):
    node = None
    for it in items:
        if it[0] == "leaf":
            sub = it[1]
        else:
            sub = None
            for t in it[1]:
                sub = t if sub is None else frozenset((sub, t))
        node = sub if node is None else frozenset((node, sub))
    return node


def canon_str(text):
    toks = re.findall(r"\(|\)|\+|l\d+", text)
    pos = 0

    def atom():
        nonlocal pos
        t = toks[pos]
        pos += 1
        if t == "(":
            v = expr()
            assert toks[pos] == ")"
            pos += 1
            return v
        return int(t[1:])

    def expr():
        nonlocal pos
        v = atom()
        while pos < len(toks) and toks[pos] == "+":
            pos += 1
            v = frozenset((v, atom()))
        return v

    return expr()


DOC = {
    "e": (
        "(l0 + (l1 + (l2 + (l3 + (l4 + (l5 + (l6 + (l7 + (l8 + (l9 + (l10 + (l11 + (l12 + (l13 + (l14 + l15)))))))))))))))",
        "(l0 + (((l1 + ((l2 + (l3 + ((l15 + l11) + l7))) + ((l14 + l10) + l6))) + ((l13 + l9) + l5)) + ((l12 + l8) + l4)))",
    ),
    "x4": (
        "(l3 + (l4 + (l5 + (l6 + (l7 + (l8 + (l9 + (l10 + (l11 + (l12 + (l13 + (l14 + l15))))))))))))",
        "(l3 + (((l4 + ((l5 + (l6 + (l14 + l10))) + (l13 + l9))) + (l12 + l8)) + ((l15 + l11) + l7)))",
    ),
    "x8": (
        "(l7 + (l8 + (l9 + (l10 + (l11 + (l12 + (l13 + (l14 + l15))))))))",
        "(l7 + (((l8 + ((l9 + (l10 + l14)) + l13)) + l12) + (l15 + l11)))",
    ),
    "x12": (
        "(l11 + (l12 + (l13 + (l14 + l15))))",
        "(l11 + ((l12 + (l13 + l14)) + l15))",
    ),
}
for y in ENTRIES:
    assert canon_items(naive_items(y)) == canon_str(DOC[y][0]), ("naive", y)
    assert canon_items(cache_items(y)) == canon_str(DOC[y][1]), ("cache", y)
    assert canon_items(naive_items(y)) != canon_items(cache_items(y)), y

out = []


def T(x, y, s, size=12, weight="normal", fill=INK, anchor="start", family=SANS):
    out.append(
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="{family}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{html.escape(s)}</text>'
    )


def R(x, y, w, h, fill, stroke, sw=1.2, rx=6, dash=None, opacity=None):
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    extra += f' fill-opacity="{opacity}"' if opacity is not None else ""
    out.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>'
    )


def para(x, y, text, size=12, fill=MUTED, maxch=235, lh=18, weight="normal"):
    words, line = text.split(" "), ""
    for w in words:
        if line and len(line) + 1 + len(w) > maxch:
            T(x, y, line, size, weight, fill)
            y += lh
            line = w
        else:
            line = f"{line} {w}" if line else w
    if line:
        T(x, y, line, size, weight, fill)
        y += lh
    return y


def fmt(entries):
    return "[" + ", ".join(entries) + "]" if entries else "[ ]"


# ---- header ----
y = 36
T(30, y, "PR 4312 · pp4 × vp4, 4 blocks × 4 layers: how the backward sums each block's gradient, naive vs rank cache", 19, "700")
y = para(30, y + 26, "Stage s runs on rank s % 4 (Interleaved1F1B), 16 stages. Stack entries: e (embedding), x4, x8, x12 (results of blocks 1, 2, 3; block 4's result goes to the head's aggregation). Every stage from an entry's commit on reads it.", 12.5)
y = para(30, y, "dℓ with index s (chips dℓ15 ... dℓ0): stage s's own contribution to an entry's gradient (its layers' residual reads). de, dx4, dx8, dx12: an entry's total gradient.", 12.5)

# ---- 1. forward payloads ----
y += 16
T(30, y, "1 · Forward P2P per hop", 15.5, "700")
X0, CW, BW, BH = 110, 320, 178, 66
GY, RH = y + 58, 98


def box_xy(s):
    return X0 + rank(s) * CW, GY + (s // P) * RH


for r in range(P):
    R(X0 + r * CW - 12, GY - 34, BW + 24, 3 * RH + BH + 46, FILL[r], "none", 0, rx=10, opacity=0.5)
    T(X0 + r * CW + BW / 2, GY - 14, f"rank {r} (one store)", 12.5, "700", STROKE[r], "middle")
for v in range(S // P):
    T(46, GY + v * RH + BH / 2 + 4, f"v{v}", 12.5, "700", MUTED, "middle")

out.append(
    '<defs><marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
    'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="#374151"/></marker></defs>'
)
for s in range(S - 1):
    x, yy = box_xy(s)
    ym = yy + BH / 2
    if rank(s) < P - 1:
        x1, x2 = x + BW + 3, x + CW - 3
        out.append(f'<line x1="{x1}" y1="{ym}" x2="{x2}" y2="{ym}" stroke="#374151" stroke-width="1.4" marker-end="url(#arr)"/>')
        mid = (x1 + x2) / 2
        T(mid, ym - 7, fmt(cache_payload(s)), 11.5, "700", INK, "middle", MONO)
        T(mid, ym + 17, fmt(naive_payload(s)), 10.5, "normal", MUTED, "middle", MONO)
    else:
        xn, yn = box_xy(s + 1)
        yg = yy + BH + 16
        out.append(
            f'<path d="M{x + BW + 3},{ym} H1296 V{yg} H74 V{yn + BH / 2} H{xn - 3}" fill="none" '
            f'stroke="#374151" stroke-width="1.2" stroke-dasharray="5 3" marker-end="url(#arr)"/>'
        )
        T(1304, ym - 4, fmt(cache_payload(s)), 11.5, "700", INK, "start", MONO)
        T(1304, ym + 13, fmt(naive_payload(s)), 10.5, "normal", MUTED, "start", MONO)
for s in range(S):
    x, yy = box_xy(s)
    r = rank(s)
    R(x, yy, BW, BH, "#ffffff", STROKE[r], 1.5, rx=7)
    T(x + 10, yy + 20, f"s{s} · {layers(s)}", 13, "700")
    if commits(s):
        T(x + 10, yy + 38, "commits " + ", ".join(commits(s)), 11.5, "700", STROKE[r])
    T(x + 10, yy + 56, "store: " + (", ".join(store_at_entry(s)) or "none"), 11, "normal", MUTED)

y = GY + 3 * RH + BH + 38
y = para(30, y, "Above each arrow, bold: what the hop sends with the cache on; below, grey: with it off (every entry on every hop). Per micro-batch: 39 entries naive, 12 cached. store: what the rank's store holds when the stage starts (cache on).", 12, MUTED)
y = para(30, y, "Backward: every hop sends the gradients of exactly the entries its forward hop sent, from s15 to s0. An entry a stage read from its rank's store sends its gradient nowhere: it is deposited in that store.", 12, MUTED)

# ---- 2. backward summation order ----
y += 20
T(30, y, "2 · Backward: in which order each entry's gradient is summed (one micro-batch)", 15.5, "700")
y = para(30, y + 22, "Left to right: the order in which the running sum grows. Under each +: the stage that performs that addition. Chip colour: the rank that computes the term.", 12, MUTED)
y = para(30, y, "Boxed: a rank's store deposit (its later readers summed, highest stage first), added at the stage that brought the entry onto that rank. Dashed chip: computed after dx12's sum, so its value already differs between the modes.", 12, MUTED)

CWp, CH, OW, GP, LX = 46, 26, 26, 6, 150


def leaf(x, cy, s):
    r = rank(s)
    R(x, cy, CWp, CH, FILL[r], STROKE[r], 1.4, rx=5, dash="4 3" if s <= 10 else None)
    out.append(
        f'<text x="{x + CWp / 2:.1f}" y="{cy + 17:.1f}" font-family="{SANS}" font-size="13" font-weight="600" '
        f'fill="{INK}" text-anchor="middle">dℓ<tspan font-size="9.5" dy="3">{s}</tspan></text>'
    )


def op(x, cy, stage):
    T(x + OW / 2, cy + 18, "+", 15, "700", INK, "middle")
    T(x + OW / 2, cy + CH + 12, f"s{stage}", 9.5, "normal", MUTED, "middle")


def row(items, cy):
    x = LX
    for i, it in enumerate(items):
        if i:
            op(x, cy, it[2])
            x += OW
        if it[0] == "leaf":
            leaf(x, cy, it[1])
            x += CWp
        else:
            dep, h = it[1], it[2]
            r = rank(h)
            gw = 2 * GP + len(dep) * CWp + (len(dep) - 1) * OW
            R(x, cy - 5, gw, CH + 21, FILL[r], STROKE[r], 1.2, rx=7, opacity=0.55)
            T(x + gw / 2, cy - 9, f"r{r} store → s{h}", 10.5, "700", STROKE[r], "middle")
            xi = x + GP
            for j, t in enumerate(dep):
                if j:
                    op(xi, cy, t)
                    xi += OW
                leaf(xi, cy, t)
                xi += CWp
            x += gw
    assert x < W - 20, x


NOTES = {
    "e": "Naive: one chain through all 16 stages. Cache: the chain runs only s3 → s2 → s1 → s0, the stages that brought e onto a rank; each rank's later readers are summed among themselves first and join as one deposit.",
    "x4": "Cache: the chain runs s6 → s5 → s4 → s3; ranks 2, 1, 0 add their deposits at s6, s5, s4 after the stage's own term; rank 3's deposit joins at s3 before s3's own term (s3 committed x4).",
    "x8": "Cache: the chain runs s10 → s9 → s8 → s7; the one-term deposits dℓ14, dℓ13, dℓ12 join at s10, s9, s8, and dℓ15 + dℓ11 at s7.",
    "x12": "Cache: dℓ15 joins at the 4th addition instead of the 1st. All five terms are bitwise the same in both modes, so dx12 is the first sum in the backward that differs.",
}
y += 14
for e in ENTRIES:
    T(30, y + 34, GRAD[e], 17, "700")
    T(92, y + 34, "naive", 12, "700", MUTED)
    row(naive_items(e), y + 16)
    T(92, y + 108, "cache", 12, "700", STROKE[3])
    row(cache_items(e), y + 90)
    y = para(LX, y + 154, NOTES[e], 11.5, MUTED, 225, 17)
    y += 14

# ---- 3. per layer ----
y += 8
T(30, y, "3 · Which layers get a different gradient", 15.5, "700")
cells = ["emb"] + [f"L{i}" for i in range(16)] + ["head"]
feeds = {"emb": "= de", "L3": "= dx4", "L7": "= dx8", "L11": "= dx12"}
cy, cw, gap = y + 16, 62, 4
for i, c in enumerate(cells):
    x = 30 + i * (cw + gap)
    differs = c == "emb" or (c.startswith("L") and int(c[1:]) <= 11)
    R(x, cy, cw, 30, "#fee2e2" if differs else "#f3f4f6", "#dc2626" if differs else "#9ca3af", 1.3, rx=5)
    T(x + cw / 2, cy + 20, c, 12.5, "700", "#991b1b" if differs else "#374151", "middle")
    if c in feeds:
        T(x + cw / 2, cy + 48, "output grad", 9.5, "normal", "#991b1b", "middle")
        T(x + cw / 2, cy + 61, feeds[c], 11, "700", "#991b1b", "middle", MONO)
lx = 30 + len(cells) * (cw + gap) + 14
R(lx, cy + 2, 14, 12, "#fee2e2", "#dc2626", 1.2, rx=3)
T(lx + 20, cy + 12, "differs, cache on vs off", 11.5, "normal", INK)
R(lx, cy + 20, 14, 12, "#f3f4f6", "#9ca3af", 1.2, rx=3)
T(lx + 20, cy + 30, "bitwise the same", 11.5, "normal", INK)
y = cy + 88
y = para(30, y, "A layer's parameter gradients come from its output gradient. The outputs of layers 3, 7 and 11 are x4, x8 and x12 (each is committed as the next layer's input), so their output gradients are exactly dx4, dx8, dx12; the embedding's is de.", 12, INK)
y = para(30, y, "Layers 12-15 and the head receive no reassociated sum, so they are bitwise the same. From layer 11 down every layer and the embedding differ, by rounding only: the same terms, added in another order.", 12, INK)
y = para(30, y + 6, "Measured on the 24-layer debug model (blocks of 12, so entries e and x12), pp4 × vp4, step-1 gradients cache on vs off: exactly layers 0-11 and tok_embeddings differ (334 of 680 parameters), layers 12-23 bitwise; max relative difference 7.4e-2 in bf16, 9.1e-6 in fp32, 1.7e-14 in fp64.", 12, INK)
y = para(30, y + 6, "Order from pipeline_stage.py (k3_pp_text 6e1f5e41c): a store reader deposits prior + grad; a stage that received the entry adds its rank's deposit after its own backward (backward_one_chunk), the stage that committed it adds it before (_retrieve_recv_grads).", 11, MUTED, 250, 16)
y = para(30, y, "Each dℓ term is itself several terms (two residual reads per layer; at s0 also the embedding's direct use). Naive adds them one by one onto the incoming sum and a store reader sums them from zero, so rows that match at stage level can still differ per read (measured at pp2 × vp2 for x12).", 11, MUTED, 250, 16)

H = int(y + 14)
svg = (
    f'<?xml version="1.0" encoding="UTF-8"?>\n<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
    f'viewBox="0 0 {W} {H}">\n<rect width="{W}" height="{H}" fill="#ffffff"/>\n' + "\n".join(out) + "\n</svg>\n"
)
with open(OUT, "w", encoding="utf-8", newline="\n") as f:
    f.write(svg)
print(OUT, W, H)
