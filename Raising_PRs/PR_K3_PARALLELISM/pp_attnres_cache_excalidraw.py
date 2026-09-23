"""Generate pp_attnres_cache.excalidraw (and a PIL preview) in the style of
torchtitan/models/common/MOE_SHARDING.md: hand-drawn entities, green boxes for
the communication primitives (solid = forward, hatched = backward), black
forward arrows, red backward arrows, orange forward labels, red backward
labels, blue annotations.

    python gen_excalidraw.py OUT.excalidraw [preview.png]
"""

import json
import math
import random
import sys
import textwrap

random.seed(4312)

BLACK = "#1e1e1e"
RED = "#e03131"
ORANGE = "#f08c00"
BLUE = "#1971c2"
GREEN = "#2f9e44"
GREEN_BG = "#b2f2bb"

FONT = 5  # Excalifont (hand-drawn); older builds fall back to Virgil
CHAR_W = 0.56  # average glyph width as a fraction of the font size

elements = []


def _id(prefix):
    return f"{prefix}-{random.randrange(16**8):08x}"


def _base(kind, x, y, w, h, **kw):
    el = {
        "id": _id(kind),
        "type": kind,
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "angle": 0,
        "strokeColor": BLACK,
        "backgroundColor": "transparent",
        "fillStyle": "solid",
        "strokeWidth": 2,
        "strokeStyle": "solid",
        "roughness": 1,
        "opacity": 100,
        "groupIds": [],
        "frameId": None,
        "index": None,
        "roundness": None,
        "seed": random.randrange(1, 2**31),
        "version": 1,
        "versionNonce": random.randrange(1, 2**31),
        "isDeleted": False,
        "boundElements": [],
        "updated": 1758585600000,
        "link": None,
        "locked": False,
    }
    el.update(kw)
    elements.append(el)
    return el


def text(x, y, s, size=16, color=BLACK, align="left", container=None, width=None):
    lines = s.split("\n")
    w = width or max(len(line) for line in lines) * size * CHAR_W
    h = len(lines) * size * 1.25
    el = _base(
        "text", x, y, w, h,
        strokeColor=color, text=s, originalText=s, fontSize=size, fontFamily=FONT,
        textAlign=align, verticalAlign="middle" if container else "top",
        containerId=container["id"] if container else None, lineHeight=1.25, autoResize=True,
    )
    if container:
        container["boundElements"].append({"id": el["id"], "type": "text"})
    return el


def box(x, y, w, h, label, kind="entity", size=14):
    style = {
        "entity": dict(strokeColor=BLACK, backgroundColor="transparent"),
        "comm_fwd": dict(strokeColor=GREEN, backgroundColor=GREEN_BG, fillStyle="solid"),
        "comm_bwd": dict(strokeColor=GREEN, backgroundColor=GREEN_BG, fillStyle="hachure"),
    }[kind]
    el = _base("rectangle", x, y, w, h, roundness={"type": 3}, **style)
    max_chars = int((w - 20) / (size * CHAR_W))
    wrapped = "\n".join(
        "\n".join(textwrap.wrap(line, max_chars)) if line else "" for line in label.split("\n")
    )
    n = wrapped.count("\n") + 1
    tw = min(w - 20, max(len(l) for l in wrapped.split("\n")) * size * CHAR_W)
    th = n * size * 1.25
    text(x + (w - tw) / 2, y + (h - th) / 2, wrapped, size, BLACK, "center", container=el, width=tw)
    return el


def anchor(el, side, frac=0.5):
    x, y, w, h = el["x"], el["y"], el["width"], el["height"]
    return {
        "t": (x + w * frac, y),
        "b": (x + w * frac, y + h),
        "l": (x, y + h * frac),
        "r": (x + w, y + h * frac),
    }[side]


def arrow(src, s_side, dst, d_side, color, label=None, *, via=(), s_frac=0.5, d_frac=0.5,
          label_at=None, size=14):
    p0 = anchor(src, s_side, s_frac)
    p1 = anchor(dst, d_side, d_frac)
    pts = [p0, *via, p1]
    rel = [[px - p0[0], py - p0[1]] for px, py in pts]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    el = _base(
        "arrow", p0[0], p0[1], max(xs) - min(xs), max(ys) - min(ys),
        strokeColor=color, points=rel, lastCommittedPoint=None,
        startBinding={"elementId": src["id"], "focus": 0, "gap": 3},
        endBinding={"elementId": dst["id"], "focus": 0, "gap": 3},
        startArrowhead=None, endArrowhead="arrow", elbowed=False,
    )
    src["boundElements"].append({"id": el["id"], "type": "arrow"})
    dst["boundElements"].append({"id": el["id"], "type": "arrow"})
    if label:
        if label_at is None:
            mid = ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2)
            label_at = (mid[0] + 8, mid[1] - 10)
        text(label_at[0], label_at[1], label, size, ORANGE if color == BLACK else RED)
    return el


# ============================================================================
# Panel 1: attn_res_cache on (the delta transport), one stage s on rank r
# ============================================================================
X = 0
text(X + 20, 10, "Kimi K3 pipeline with attn_res_cache on: stage s on rank r", 22)
text(X + 20, 42, "a hop carries (hidden, delta), delta = the blocks the receiving rank has not seen;"
     " P2P only, no collective", 15, BLUE)
text(X + 20, 64, "the routing tables (BlockLayoutTables) are a pure function of the split,"
     " the same on every rank: commits_at, cache_at_entry, delta_to_send, deposits_expected", 15, BLUE)

MX = X + 330  # middle (forward) column
RX = X + 740  # right (backward) column
CW = 250

# --- forward chain, top to bottom -------------------------------------------
text(MX, 118, "input activations from stage s-1\nhidden [T, D], delta [T, Nd, D]", 15, ORANGE)
recv = box(MX, 168, CW, 44, "P2P recv (hidden, delta)", "comm_fwd")
assemble = box(MX, 296, CW, 76,
               "assemble_stack\nstack leaf [T, N, D] = store blocks + delta columns, in block order", "entity", 13)
model = box(MX, 436, CW, 84,
            "stage s model part: layers l..m\n(a block's first layer cats its input hidden into the stack)", "entity", 13)
pack = box(MX, 584, CW, 64, "pack_outgoing_delta\nthe columns the next rank lacks (delta_to_send)", "entity", 13)
send = box(MX, 712, CW, 44, "P2P send (hidden', delta')", "comm_fwd")
text(MX, 770, "output to stage s+1\nhidden' [T, D], delta' [T, Nd', D]", 15, ORANGE)
text(MX, 836, "FORWARD", 20, ORANGE)

arrow(recv, "b", assemble, "t", BLACK, "hidden, delta", label_at=(MX + 133, 240))
arrow(assemble, "b", model, "t", BLACK, "hidden, stack", label_at=(MX + 133, 390))
arrow(model, "b", pack, "t", BLACK, "hidden', stack' [T, N', D]", label_at=(MX + 133, 538))
arrow(pack, "b", send, "t", BLACK, "hidden', delta'", label_at=(MX + 133, 690))

# --- the rank store (left) ----------------------------------------------------
store = box(X + 20, 296, 210, 150,
            "rank store (PPRankLocalCache)\nthe blocks committed at stages <= s-P, keyed (micro-batch, block),"
            " and the gradient deposits\none per rank, shared by its stages", "entity", 13)
text(X + 20, 700, "release: after the rank's last stage\nforward for a micro-batch, its\nblocks leave the store", 13, BLUE)

arrow(store, "r", assemble, "l", BLACK, "held blocks", s_frac=0.22, d_frac=0.3,
      label_at=(X + 240, 304), size=13)
arrow(assemble, "l", store, "r", BLACK, "cache the delta columns", s_frac=0.85, d_frac=0.6,
      label_at=(X + 234, 392), size=13)
arrow(model, "l", store, "r", BLACK, "commit the new blocks", s_frac=0.5, d_frac=0.9,
      via=((X + 280, 478),), label_at=(X + 150, 486), size=13)

# --- backward chain, bottom to top (right column) ---------------------------
text(RX, 770, "activation gradient from stage s+1\ngrad hidden' [T, D], grad delta' [T, Nd', D]", 15, RED)
recv_g = box(RX, 712, CW + 20, 44, "P2P recv grad (hidden', delta')", "comm_bwd")
retrieve = box(RX, 584, CW + 20, 64,
               "_retrieve_recv_grads\n+ the deposits for the blocks this stage committed", "entity", 13)
split = box(RX, 296, CW + 20, 76,
            "split_stack_grad\nreceived columns -> grad delta (wire order); stored columns -> deposits",
            "entity", 13)
send_g = box(RX, 168, CW + 20, 44, "P2P send grad (hidden, delta)", "comm_bwd")
text(RX, 112, "to stage s-1: grad hidden [T, D], grad delta [T, Nd, D]\n(whether delta needs a gradient is read off"
     " the receive metadata)", 13, RED)
text(RX, 836, "BACKWARD", 20, RED)

arrow(recv_g, "t", retrieve, "b", RED, "grad hidden', grad delta'", label_at=(RX + 143, 668))
arrow(retrieve, "l", model, "r", RED, "backward through\nthe stage", s_frac=0.5, d_frac=0.78,
      label_at=(X + 622, 592), size=13)
arrow(model, "r", split, "l", RED, "grad hidden,\ngrad stack leaf [T, N, D]", s_frac=0.25, d_frac=0.55,
      label_at=(X + 596, 470), size=13)
arrow(split, "t", send_g, "b", RED, "grad hidden, grad delta", label_at=(RX + 143, 240))

# store <-> backward column: the deposits go in from split and come out at retrieve
arrow(split, "l", store, "t", RED, "deposit the stored columns' gradient",
      s_frac=0.18, d_frac=0.5, via=((MX + CW + 60, 310), (MX + CW + 60, 226), (X + 125, 226)),
      label_at=(X + 596, 232), size=13)
arrow(store, "b", retrieve, "b", RED, "collect the deposits (deposits_expected of them, else raise)",
      s_frac=0.5, d_frac=0.15, via=((X + 125, 676), (RX + 40, 676)),
      label_at=(X + 232, 656), size=13)

text(RX, 440, "the last stage sends nothing: it runs the\naggregation (output_res_proj, output_res_norm,\nlm_head) on (hidden, stack)", 13, BLUE)

# ============================================================================
# Panel 2: attn_res_cache off (the whole stack on every hop)
# ============================================================================
X2 = 1120
text(X2 + 20, 10, "attn_res_cache off: the whole stack every hop", 22)
text(X2 + 20, 42, "the same stage with no store and no deposits; the payload grows with the stage index", 15, BLUE)
text(X2 + 20, 64, "plain 1F1B, one stage per rank, is this transport by construction", 15, BLUE)

M2 = X2 + 60
R2 = X2 + 400
text(M2, 118, "input activations from stage s-1\nhidden [T, D], stack [T, N, D]", 15, ORANGE)
recv2 = box(M2, 168, CW, 44, "P2P recv (hidden, stack)", "comm_fwd")
model2 = box(M2, 436, CW, 84, "stage s model part: layers l..m\n(the stack is the leaf; nothing is assembled)", "entity", 13)
send2 = box(M2, 712, CW, 44, "P2P send (hidden', stack')", "comm_fwd")
text(M2, 770, "output to stage s+1\nhidden' [T, D], stack' [T, N', D]", 15, ORANGE)
text(M2, 836, "FORWARD", 20, ORANGE)
arrow(recv2, "b", model2, "t", BLACK, "hidden, stack", label_at=(M2 + 133, 310))
arrow(model2, "b", send2, "t", BLACK, "hidden', stack'", label_at=(M2 + 133, 600))

text(R2, 770, "activation gradient from stage s+1\ngrad hidden', grad stack'", 15, RED)
recv2_g = box(R2, 712, CW, 44, "P2P recv grad (hidden', stack')", "comm_bwd")
send2_g = box(R2, 168, CW, 44, "P2P send grad (hidden, stack)", "comm_bwd")
text(R2, 118, "to stage s-1: grad hidden, grad stack", 15, RED)
text(R2, 836, "BACKWARD", 20, RED)
arrow(recv2_g, "t", model2, "r", RED, "backward through the stage", d_frac=0.75,
      via=((R2 + 125, 499),), label_at=(R2 - 90, 560), size=13)
arrow(model2, "r", send2_g, "b", RED, "grad hidden, grad stack", s_frac=0.25,
      via=((R2 + 125, 457),), label_at=(R2 + 135, 300), size=13)

# ============================================================================
doc = {
    "type": "excalidraw",
    "version": 2,
    "source": "https://excalidraw.com",
    "elements": elements,
    "appState": {"gridSize": 20, "viewBackgroundColor": "#ffffff"},
    "files": {},
}
out = sys.argv[1]
with open(out, "w", encoding="utf-8") as f:
    json.dump(doc, f, indent=2)
print(f"{len(elements)} elements -> {out}")

# ---------------------------------------------------------------- preview ---
if len(sys.argv) > 2:
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1860, 900
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    cache = {}

    def fnt(size):
        if size not in cache:
            try:
                cache[size] = ImageFont.truetype("arial.ttf", int(size * 0.9))
            except Exception:
                cache[size] = ImageFont.load_default()
        return cache[size]

    for el in elements:
        if el["type"] == "rectangle":
            fill = el["backgroundColor"] if el["backgroundColor"] != "transparent" else None
            d.rounded_rectangle([el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"]],
                                radius=10, outline=el["strokeColor"], fill=fill, width=2)
    for el in elements:
        if el["type"] == "arrow":
            pts = [(el["x"] + px, el["y"] + py) for px, py in el["points"]]
            d.line(pts, fill=el["strokeColor"], width=3)
            (x0, y0), (x1, y1) = pts[-2], pts[-1]
            a = math.atan2(y1 - y0, x1 - x0)
            for da in (math.pi / 7, -math.pi / 7):
                d.line([(x1, y1), (x1 - 14 * math.cos(a - da), y1 - 14 * math.sin(a - da))],
                       fill=el["strokeColor"], width=3)
    for el in elements:
        if el["type"] == "text":
            d.multiline_text((el["x"], el["y"]), el["text"], fill=el["strokeColor"],
                             font=fnt(el["fontSize"]), align=el["textAlign"])
    img.save(sys.argv[2])
    print("preview ->", sys.argv[2])
