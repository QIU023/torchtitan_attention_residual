"""Planner cost ratios for (seq, image side, images-per-sample cap) from microbench outputs, to pick levels in the K3
report's range (0.1 to 0.3).

Usage: python choose_levels_k3range.py <MANIFEST.json> <microbench output at seq A> [<microbench output at seq B> ...]

cost ratio (vision_dep.bubble_cost_ratio) = mean images per image-carrying micro-batch x one image's encode forward,
over the split's mean middle stage forward at that seq; this is choose_levels.py's definition, for every image side the
microbench measured instead of 1024 px only. The stage forward of a seq that was not measured is scaled linearly from
the longest measured one; the encode time is taken from the measured seq closest to it (it does not depend on seq).
"""

import json
import re
import sys

SIDES = {224: 224, 448: 448, 768: 756, 1024: 1008}
TOKENS = {px: (side // 14 // 2) ** 2 for px, side in SIDES.items()}


def read(path):
    text = open(path).read()
    m = re.search(r"dim (\d+) seq (\d+): text stage forward ([\d.]+) ms", text)
    enc = {int(px): float(f) for px, f in re.findall(r"\| (\d+) \| \d+ \| ([\d.]+) \| [\d.]+ \|", text)}
    return int(m.group(2)), float(m.group(3)), enc, int(m.group(1))


def main():
    counts = json.load(open(sys.argv[1]))["n_images_counts"]
    carrying = sum(counts[1:])
    runs = sorted(read(p) for p in sys.argv[2:])
    dim = runs[0][3]
    longest_seq, longest_stage = runs[-1][0], runs[-1][1]
    print(f"dim {dim}; measured stage forward: " + ", ".join(f"seq {s} {f:.2f} ms" for s, f, _, _ in runs))
    print("| seq | stage forward ms | image px | tokens per image | cap | images per carrying micro-batch | "
          "encode forward ms (one image) | cost ratio |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for seq in sorted({s for s, _, _, _ in runs} | {8192}):
        measured = [r for r in runs if r[0] == seq]
        stage = measured[0][1] if measured else longest_stage * seq / longest_seq
        enc = min(runs, key=lambda r: abs(r[0] - seq))[2]
        for px in sorted(enc):
            for cap in (1, 2, 3):
                if cap * TOKENS[px] > seq:
                    continue
                mean = sum(min(n, cap) * c for n, c in enumerate(counts)) / carrying
                ratio = mean * enc[px] / stage
                mark = " <-" if 0.1 <= ratio <= 0.3 else ""
                print(f"| {seq} | {stage:.2f}{'' if measured else ' (scaled)'} | {px} | {TOKENS[px]} | {cap} | "
                      f"{mean:.3f} | {enc[px]:.2f} | {ratio:.3f}{mark} |")


if __name__ == "__main__":
    main()
