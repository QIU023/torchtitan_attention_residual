"""#4380 tower benchmark table from tower_bench.py's results.jsonl, logbook kit only: per-GPU peak allocated GiB and
median step ms, main at CP 1/2/4 and the PR at CP 2/4. usage: python tower_table.py results.jsonl [cells OOM in their logs]"""

import json
import sys

rows = [json.loads(line) for line in open(sys.argv[1])]
# cells whose ranks died of CUDA OOM before writing a record, read from their logs: tower_table.py results.jsonl [oom names]
oom_logged = set(sys.argv[2:])
by = {(r["ac"], r["case"], "pr" if r["tree"] == "tt_cpmm" else "main", r["cp"]): r for r in rows}
cols = [("main", 1), ("main", 2), ("main", 4), ("pr", 2), ("pr", 4)]
cases = list(dict.fromkeys(r["case"] for r in rows))


def cell(r, key, oom_logged=False):
    if r is None:
        return "OOM (log)" if oom_logged else "-"
    if r["status"] != "ok":
        return "OOM"
    return f"{r[key]:.1f}" if key == "max_peak_gib" else f"{r[key]:.0f}"


for ac in dict.fromkeys(r["ac"] for r in rows):
    for key, label in (("max_peak_gib", "peak GiB per GPU"), ("ms", "ms per forward + backward")):
        print(f"\nAC {ac}: {label}\n")
        print("| case | patches | " + " | ".join(f"{t} CP{c}" for t, c in cols) + " |")
        print("|---|---|" + "---|" * len(cols))
        for case in cases:
            any_row = next((by[k] for k in by if k[0] == ac and k[1] == case), None)
            if any_row is None:
                continue
            print(
                f"| {case} | {any_row['patches']} | "
                + " | ".join(
                    cell(by.get((ac, case, t, c)), key, f"{t}_{ac}_{case}_cp{c}" in oom_logged)
                    for t, c in cols
                )
                + " |"
            )
