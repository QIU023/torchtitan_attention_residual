"""Block-only account of the 09-30 H100 runs, recorded with the old probe (a forward send's op 0, the hidden
state, counted as a block whenever it was held by nothing else): python h100_blocks_from_old.py [<results dir>]

Every hidden state is one [T, D] bf16 tensor, and the old probe counted every pending hidden send (its storage was
never in the other sets). So, per action of traced step 5: derived blocks = old block_gib - pending hidden sends,
where a hidden send is issued after each forward of a stage that is not the last one and is pending until
- #4656: the end of the step (torch's action-list runtime waits every send there);
- PR A: that stage's backward of the same micro-batch.
The 10-01 5060 recheck with the fixed probe measures both directly and checks this count. For PR A the derived
blocks also equal old block_gib - fwd_sends_only_gib when no block send is ever held by its send alone.
Units are [T, D] bf16 blocks (2048 x dim x 2 bytes). The tight bound frees each block at the backward of the stage
that brought it onto the rank; the paper bound keeps a micro-batch's blocks until the rank's last backward of it.
"""

import json
import os
import re
import sys

ACTION = re.compile(r"^(\d+)([FB])(\d+)$")
RUNS = [("pra_h100", 6144, "pp4vp2", 8), ("pra_h100", 6144, "pp4vp4", 16), ("pra_h100_d5120", 5120, "pp2vp2", 4),
        ("pra_h100_d5120", 5120, "dp2pp2vp2", 4), ("pra_h100_d5120", 5120, "pp2vp4", 8)]
# With the fixed probe (records carry hidden_sends_alive_gib), e.g. the 5060 recheck:
#   python h100_blocks_from_old.py <dir> .:2048:pp4vp2:8
# block_gib then already leaves the hidden sends out; the table also checks the measured pending hidden sends
# against the count above, and that PR A's fwd_sends_only_gib (block sends held by nothing else) stays 0.


def pending_hidden(actions, last, tree):
    out, issued, waited = [], [], set()
    for a in actions:
        stage, kind, mb = ACTION.match(a["action"]).groups()
        stage, mb = int(stage), int(mb)
        if kind == "B" and tree == "pra":
            waited.add((stage, mb))
        out.append(sum(1 for k in issued if k not in waited))
        if kind == "F" and stage != last:
            issued.append((stage, mb))
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "kit_h100_2026-09-30", "results")
    runs = RUNS
    if len(sys.argv) > 2:
        runs = [(f, int(dim), L, int(n)) for f, dim, L, n in (x.split(":") for x in sys.argv[2:])]
    print("| layout | dim | tree | rank | actions | derived = tight | derived - tight, min / max | derived max | tight max | paper max | old max | hidden pending, max | check (old probe, PR A: old - sends only = derived; fixed probe: measured hidden = count, block sends only) |")
    print("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for folder, dim, L, stages in runs:
        blk = 2048 * dim * 2 / 2**30
        for tree in ("4656", "pra"):
            mem = os.path.join(root, folder, f"{tree}_{L}", "mem")
            if not os.path.isdir(mem):
                continue
            for name in sorted(os.listdir(mem)):
                d = json.load(open(os.path.join(mem, name)))
                acts = d.get("actions", [])
                if not acts:
                    continue
                hidden = pending_hidden(acts, stages - 1, tree)
                fixed = "hidden_sends_alive_gib" in acts[0]
                old = [a["block_gib"] / blk + (a["hidden_sends_alive_gib"] / blk if fixed else 0.0) for a in acts]
                derived = [a["block_gib"] / blk for a in acts] if fixed else [o - h for o, h in zip(old, hidden)]
                tight = [a["tight_bound_gib"] / blk for a in acts]
                diff = [x - t for x, t in zip(derived, tight)]
                eq = sum(1 for x in diff if abs(x) < 0.05)
                check = "-"
                if fixed:
                    measured = [a["hidden_sends_alive_gib"] / blk for a in acts]
                    check = (f"hidden = count: {all(abs(m - h) < 0.05 for m, h in zip(measured, hidden))}; "
                             f"block sends only, max {max(a['fwd_sends_only_gib'] for a in acts) / blk:.2f}")
                elif tree == "pra":
                    alt = [(a["block_gib"] - a["fwd_sends_only_gib"]) / blk for a in acts]
                    check = str(all(abs(x - y) < 0.05 for x, y in zip(alt, derived)))
                print(
                    f"| {L} | {dim} | {tree} | {d['rank']} | {len(acts)} | {eq} | {min(diff):.2f} / {max(diff):.2f} | "
                    f"{max(derived):.2f} | {max(tight):.2f} | {max(a['paper_bound_gib'] for a in acts) / blk:.2f} | "
                    f"{max(old):.2f} | {max(hidden)} | {check} |"
                )


if __name__ == "__main__":
    main()
