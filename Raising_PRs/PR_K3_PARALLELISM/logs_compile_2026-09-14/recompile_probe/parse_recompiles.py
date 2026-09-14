import re, sys, collections
cats = [("MLA/KDA module structure", r"KeyError on .*_modules\['(attention|delta_attention)'\]"),
        ("block-opening attribute (layer_id % block)", r"layer_id %"),
        ("stack-width dim (args[1] dim 1)", r"'args\[1\]' size mismatch at index 1"),
        ("tower image shape", r"'args\[0\]' size mismatch|mask_mod"),
        ("wrapper type", r"check_type_id"),
        ("call arity (tower vs decoder)", r"len\(args\) =="),]
for path in sys.argv[1:]:
    txt = open(path).read().splitlines()
    events = []; cur = None
    for l in txt:
        m = re.search(r"\[(\d+)/(\d+)\] \[__recompiles\] Recompiling function (\w+) in (\S+)", l)
        if m:
            cur = {"frame": m.group(1), "idx": int(m.group(2)), "fn": m.group(3), "file": m.group(4), "fails": []}; events.append(cur); continue
        m = re.search(r"__recompiles\]     - (\d+)/(\d+): (.*)", l)
        if m and cur is not None:
            cur["fails"].append(m.group(3))
    wr = [e for e in events if "checkpoint_wrapper" in e["file"]]
    nvar = (max(e["idx"] for e in wr) + 1) if wr else 1
    print(f"== {path.split('/')[-1]}: checkpoint_wrapper.forward compiled variants = {nvar} ({len(wr)} recompiles)")
    for e in wr:
        c = sorted({name for name, rx in cats for f in e["fails"] if re.search(rx, f)})
        print(f"   variant {e['idx']}: new because it failed, against the existing entries: {', '.join(c) or '?'}")
    tally = collections.Counter(name for e in wr for name, rx in cats if any(re.search(rx, f) for f in e["fails"]))
    print("   reasons over all recompiles:", dict(tally))
    others = collections.Counter(e["fn"] for e in events if "checkpoint_wrapper" not in e["file"])
    print("   other frames recompiled:", dict(others))


def per_frame(path):
    import re, collections
    top = collections.defaultdict(int)
    for l in open(path):
        m = re.search(r"\[(\d+)/(\d+)\] \[__recompiles\] Recompiling function (\w+) in (\S+)", l)
        if m:
            key = f"{m.group(3)} @ {m.group(4).split('/')[-1]}"
            top[key] = max(top[key], int(m.group(2)) + 1)
    return dict(top)


if __name__ == "__main__" and len(sys.argv) > 1:
    for p in sys.argv[1:]:
        print(f"   per-frame compiled variants ({p.split('/')[-1]}): {per_frame(p)}")
