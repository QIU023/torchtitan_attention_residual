"""Remap assign.py's engine-file table from the kit's HEAD to a new head: python remap.py <new head> [default ordinal].

Every old item is mapped to its new index by a text alignment of the two item lists; a
replaced line inherits the old item's ordinal; a genuinely new item takes the default
ordinal and is listed for review. VARIANTS keys and MOVES indices follow the same map.
The result is printed as Python (RANGES, VARIANT_KEYS, MOVES) and written to remap_table.py.
"""
import difflib
import pprint
import sys

import assign as A
import splitlib as S

NEW = sys.argv[1]
DEFAULT = sys.argv[2] if len(sys.argv) > 2 else "01"
IMPL = A.IMPL

old_items, _, _ = S.build_items(IMPL)
old_ord = {}
for (lo, hi), name in A.ASSIGN[IMPL]:
    for i in range(lo, hi + 1):
        if old_items[i].kind != "ctx":
            old_ord[i] = name
S.HEAD = NEW
new_items, _, _ = S.build_items(IMPL)


def key(it):
    return (it.kind, it.revs[0][1] if it.kind != "add" else it.revs[1][1])


sm = difflib.SequenceMatcher(None, [key(i) for i in old_items], [key(i) for i in new_items], autojunk=False)
old2new, replaced, new_only, old_gone = {}, [], [], []
for tag, i1, i2, j1, j2 in sm.get_opcodes():
    if tag == "equal":
        for d in range(i2 - i1):
            old2new[i1 + d] = j1 + d
    elif tag == "replace":
        for d in range(max(i2 - i1, j2 - j1)):
            oi, nj = i1 + d, j1 + d
            if oi < i2 and nj < j2:
                old2new[oi] = nj
                replaced.append((oi, nj))
            elif nj < j2:
                new_only.append(nj)
            else:
                old_gone.append(oi)
    elif tag == "insert":
        new_only.extend(range(j1, j2))
    elif tag == "delete":
        old_gone.extend(range(i1, i2))

new_ord = {}
for oi, nj in old2new.items():
    if oi in old_ord:
        if new_items[nj].kind == "ctx":
            print(f"WARN old changed item {oi} maps to a ctx item {nj}")
            continue
        new_ord[nj] = old_ord[oi]
unassigned = [nj for nj in new_only if new_items[nj].kind != "ctx"]
for nj in unassigned:
    new_ord[nj] = DEFAULT
# a base line the new head edits pairs with an old ctx item and carries no ordinal: new work
missing = [it.idx for it in new_items if it.kind != "ctx" and it.idx not in new_ord]
for nj in missing:
    new_ord[nj] = DEFAULT
    unassigned.append(nj)
unassigned.sort()

ranges = []
for nj in sorted(new_ord):
    name = new_ord[nj]
    if ranges and ranges[-1][1] == name and all(
        new_items[k].kind == "ctx" or new_ord.get(k) == name for k in range(ranges[-1][0][1] + 1, nj)
    ):
        ranges[-1] = ((ranges[-1][0][0], nj), name)
    else:
        ranges.append(((nj, nj), name))

variant_keys = {}
for k in A.VARIANTS.get(IMPL, {}):
    if k not in old2new:
        print(f"WARN variant key {k} has no counterpart on the new head")
        continue
    variant_keys[k] = old2new[k]
moves = []
for (lo, hi), where, anchor in A.MOVES.get(IMPL, []):
    moves.append(((old2new[lo], old2new[hi]), where, old2new[anchor]))

print(f"old items {len(old_items)} -> new items {len(new_items)}; matched {len(old2new)}, replaced {len(replaced)}, old gone {len(old_gone)}, new only {len(new_only)} ({len(unassigned)} changed, -> {DEFAULT})")
for oi, nj in replaced:
    if oi in old_ord:
        print(f"  replaced {oi}->{nj} [{old_ord[oi]}]: {key(old_items[oi])[1]!r:.70} -> {key(new_items[nj])[1]!r:.70}")
for nj in unassigned:
    print(f"  new {nj} [{DEFAULT}] {new_items[nj].kind}: {key(new_items[nj])[1]!r:.90}")
for oi in old_gone:
    if oi in old_ord:
        print(f"  gone {oi} [{old_ord[oi]}]: {key(old_items[oi])[1]!r:.80}")
with open("remap_table.py", "w") as f:
    f.write("RANGES = " + pprint.pformat(ranges) + "\n")
    f.write("VARIANT_KEYS = " + pprint.pformat(variant_keys) + "\n")
    f.write("MOVES = " + pprint.pformat(moves) + "\n")
print("ranges:", len(ranges), "variant keys:", variant_keys, "moves:", moves)
