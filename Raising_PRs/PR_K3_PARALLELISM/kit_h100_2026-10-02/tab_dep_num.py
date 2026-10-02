"""The DEP numerics table from dep_new_numerics.sh: loss / grad norm at steps 1, 10, 50 and 100, and how many of the
100 steps each cell prints identically to DEP off (and bubble to K2.5).

Usage: python tab_dep_num.py <dep_new_numerics results dir>
"""

import os
import sys

from cmp_steps import parse


def main():
    root = sys.argv[1]
    cells = [("DEP off", "num_off_a"), ("DEP off again", "num_off_b"), ("K2.5 form", "num_k25"),
             ("`bubble`", "num_bubble_new")]
    runs = {name: parse(os.path.join(root, d, "run.log")) for name, d in cells}
    ref = runs["DEP off"]
    print("| cell | step 1 | step 10 | step 50 | step 100 | steps identical to DEP off |")
    print("|---|---:|---:|---:|---:|---:|")
    for name, _ in cells:
        rows = runs[name]
        same = sum(rows.get(s) == ref.get(s) for s in ref)
        vals = " | ".join(f"{rows[s][0]} / {rows[s][1]}" if s in rows else "-" for s in (1, 10, 50, 100))
        print(f"| {name} | {vals} | {'' if name == 'DEP off' else f'{same} / {len(ref)}'} |")
    k25, bub = runs["K2.5 form"], runs["`bubble`"]
    first = next((s for s in sorted(ref) if k25.get(s) != ref.get(s)), None)
    print(f"bubble vs K2.5: {sum(bub.get(s) == k25.get(s) for s in k25)} / {len(k25)} identical; "
          f"K2.5 first differs from DEP off at step {first}")


if __name__ == "__main__":
    main()
