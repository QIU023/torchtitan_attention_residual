"""Markdown table for the AC-reuse PR body from an H200 run directory (run_ac_table.sh / run_ac_ci_cell.sh output).

    python ac_table_md.py <dir> [reference_cell]

Per cell: step-1 and step-10 loss / grad norm, the count of steps whose loss and grad norm equal the
reference cell's, peak memory (max of the per-step max-reserved figure), tps averaged over steps 6-10.
"""
import os
import re
import sys

LINE = re.compile(r"step: *(\d+) loss: *([0-9.]+) grad_norm: *([0-9.]+) memory: *([0-9.]+)GiB\([0-9.]+%\) tps: *([0-9,]+)")


def read(d):
    rows = {}
    p = os.path.join(d, "steps.txt")
    if not os.path.isfile(p):
        return rows
    for ln in open(p):
        m = LINE.search(ln)
        if m:
            step, mem, tps = int(m.group(1)), float(m.group(4)), int(m.group(5).replace(",", ""))
            if step in rows:  # one line per rank: keep the loss of the first, the max memory, the max tps
                prev = rows[step]
                rows[step] = (prev[0], prev[1], max(prev[2], mem), max(prev[3], tps))
            else:
                rows[step] = (m.group(2), m.group(3), mem, tps)
    return rows


def main():
    root = sys.argv[1]
    ref_name = sys.argv[2] if len(sys.argv) > 2 else "main_none"
    cells = {n: read(os.path.join(root, n)) for n in sorted(os.listdir(root)) if os.path.isdir(os.path.join(root, n)) and not n.startswith(("warm_", "cache"))}
    ref = cells.get(ref_name, {})
    print("| cell | step 1 loss / grad norm | step 10 loss / grad norm | steps equal to " + ref_name + " (loss and grad norm) | peak memory (rank 0, max reserved) | tps (steps 6 to 10) |")
    print("| --- | --- | --- | ---: | ---: | ---: |")
    for n, rows in cells.items():
        if not rows:
            print(f"| {n} | (no steps) | | | | |")
            continue
        last = max(rows)
        eq = sum(1 for s in rows if s in ref and rows[s][:2] == ref[s][:2])
        tps = [rows[s][3] for s in rows if 6 <= s <= 10]
        s1, s10 = rows.get(1), rows.get(10, rows[last])
        print(f"| {n} | `{s1[0]}` / `{s1[1]}` | `{s10[0]}` / `{s10[1]}` | {('reference' if n == ref_name else f'{eq} / {len(rows)}')} | {max(r[2] for r in rows.values()):.2f} GiB | {round(sum(tps) / len(tps)) if tps else 'n/a'} |")
    # per-step detail for the logbook
    print()
    print("per step (loss/grad norm; memory; tps):")
    for n, rows in cells.items():
        print(f"  {n}: " + "  ".join(f"{s}:{rows[s][0]}/{rows[s][1]};{rows[s][2]:.2f};{rows[s][3]}" for s in sorted(rows)))


if __name__ == "__main__":
    main()
