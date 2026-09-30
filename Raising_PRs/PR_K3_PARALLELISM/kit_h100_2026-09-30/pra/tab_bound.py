"""Tables for run_pra_h100.sh (09-30 H100 copy): python tab_bound.py <results dir> <layout> [<layout> ...]

Per layout: the steps whose loss and grad norm match between #4656 and PR A; every rank's peak allocated at
step 10 (not the traced step 5); and, in the traced step, every rank's block memory at the action where its
allocated memory peaks, next to the AttnRes bound at that point ("tight": each block freed at the backward
of the stage that brought it; "paper": each block of a micro-batch kept until its last backward on the rank).
"""

import json
import os
import re
import sys

STEP = re.compile(r"step:\s+(\d+)\s+loss:\s+(-?[\d.]+)\s+grad_norm:\s+([\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def losses(path):
    out = {}
    if not os.path.exists(path):
        return out
    for line in ANSI.sub("", open(path, errors="replace").read()).splitlines():
        m = STEP.search(line)
        if m and float(m.group(2)) > 0:
            out.setdefault(int(m.group(1)), (m.group(2), m.group(3)))
    return out


def ranks(root, run):
    folder = os.path.join(root, run, "mem")
    out = {}
    if not os.path.isdir(folder):
        return out
    for name in sorted(os.listdir(folder)):
        if name.startswith("rank") and name.endswith(".json"):
            d = json.load(open(os.path.join(folder, name)))
            out[d["rank"]] = d
    return out


def main():
    root, layouts = sys.argv[1], sys.argv[2:]
    for L in layouts:
        na, nb = os.environ.get("NA", "4656"), os.environ.get("NB", "pra")
        a, b = f"{na}_{L}", f"{nb}_{L}"
        la, lb = losses(os.path.join(root, a, "run.log")), losses(os.path.join(root, b, "run.log"))
        same = sum(1 for s in la if lb.get(s) == la[s])
        rc = [open(os.path.join(root, r, "rc")).read().strip() if os.path.exists(os.path.join(root, r, "rc")) else "-" for r in (a, b)]
        print(f"\n## {L}: {a} {rc[0]}, {b} {rc[1]}; steps equal (loss and grad norm) {same} / {len(la)}")
        cols = [s for s in (1, 10, 50, 100) if s in la]
        if la:
            print("| tree | " + " | ".join(f"step {s} loss / grad norm" for s in cols) + " |")
            print("|---|" + "---:|" * len(cols))
            for name, rec in ((na, la), (nb, lb)):
                print(f"| {name} | " + " | ".join(f"{rec[s][0]} / {rec[s][1]}" if s in rec else "-" for s in cols) + " |")
        ra, rb = ranks(root, a), ranks(root, b)
        print("\n| rank | peak allocated step 10, #4656 | PR A | saved | max over steps 2 on, #4656 | PR A | saved |")
        print("|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(ra):
            reca, recb = ra[r]["records"], rb.get(r, {}).get("records", [])
            pa = reca[10]["max_allocated_gib"] if len(reca) > 10 else float("nan")
            pb = recb[10]["max_allocated_gib"] if len(recb) > 10 else float("nan")
            ma = max((x["max_allocated_gib"] for x in reca[2:]), default=float("nan"))
            mb = max((x["max_allocated_gib"] for x in recb[2:]), default=float("nan"))
            print(f"| {r} | {pa:.2f} | {pb:.2f} | {pa - pb:.2f} | {ma:.2f} | {mb:.2f} | {ma - mb:.2f} |")
        print("\nTraced step 5, each rank at the action where allocated memory peaks (GiB):")
        print("| rank | tree | action | allocated | blocks held | store | stage inputs | stage outputs | sends only | recv buffers | bound tight | bound paper | blocks max over step | tight max | paper max |")
        print("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(ra):
            for tree, rr in ((a.rsplit("_", 1)[0], ra), (b.rsplit("_", 1)[0], rb)):
                acts = rr.get(r, {}).get("actions", [])
                if not acts:
                    continue
                pk = max(acts, key=lambda x: x["peak_gib"])
                bmax = max(x["block_gib"] for x in acts)
                tmax = max(x["tight_bound_gib"] for x in acts)
                pmax = max(x["paper_bound_gib"] for x in acts)
                print(
                    f"| {r} | {tree} | {pk['action']} | {pk['peak_gib']:.2f} | {pk['block_gib']:.2f} | {pk['store_gib']:.2f} | "
                    f"{pk['stage_inputs_gib']:.2f} | {pk['stage_outputs_gib']:.2f} | {pk['fwd_sends_only_gib']:.2f} | "
                    f"{pk['recv_buffers_gib']:.2f} | {pk['tight_bound_gib']:.2f} | {pk['paper_bound_gib']:.2f} | "
                    f"{bmax:.2f} | {tmax:.2f} | {pmax:.2f} |"
                )


if __name__ == "__main__":
    main()
