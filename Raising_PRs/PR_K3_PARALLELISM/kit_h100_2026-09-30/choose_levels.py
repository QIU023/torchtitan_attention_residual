"""Vision share of a DEP step and the planner's cost ratio for (seq, images-per-sample cap) settings, from measured times.

Usage: python choose_levels.py <microbench output> <MANIFEST.json> [--ac full]

Inputs: microbench_ratio.py's output (per layer forward and backward at seq 2048, the mean middle stage of the split it
was run for, encode forward and forward + backward per 1024 px image) and the dataset MANIFEST (n_images_counts).
A micro-batch is one sample padded to seq (the multimodal loader does not pack), so the text side costs seq tokens
whatever the image count; text time scales linearly with seq from 2048 (the MLA layers' attention grows faster, so
long seqs slightly overstate the vision share).
  vision share   mean images per micro-batch x (encode forward + recompute and backward) against the whole text model
                 (the sum of every layer's forward and backward, plus one more forward per layer with --ac full);
                 independent of the parallel layout
  cost ratio     vision_dep.bubble_cost_ratio: mean images per image-carrying micro-batch x encode forward, in units of
                 the split's mean middle stage forward at that seq; depends on the split the microbench ran for
"""

import json
import re
import sys

PX, TOKENS = 1024, (1008 // 14 // 2) ** 2


def _floats(text, label):
    return [float(x) for x in re.search(rf"{label}: ([\d. ]+)", text).group(1).split()]


def main():
    text = open(sys.argv[1]).read()
    full_ac = "--ac" in sys.argv and sys.argv[sys.argv.index("--ac") + 1] == "full"
    stages = int(re.search(r"stages (\d+), middle stage layers", text).group(1))
    stage_f = float(re.search(r"text stage forward ([\d.]+) ms", text).group(1))
    fwd, bwd = _floats(text, "per layer forward ms"), _floats(text, "per layer backward ms")
    model_2048 = sum(fwd) + sum(bwd) + (sum(fwd) if full_ac else 0.0)
    row = re.search(rf"\| {PX} \| \d+ \| ([\d.]+) \| ([\d.]+) \|", text)
    enc_f, enc_fb = float(row.group(1)), float(row.group(2))
    counts = json.load(open(sys.argv[2]))["n_images_counts"]
    total = sum(counts)
    print(f"measured at seq 2048: {len(fwd)} layers forward {sum(fwd):.2f} / backward {sum(bwd):.2f} ms, "
          f"text model per micro-batch {model_2048:.2f} ms ({'full AC' if full_ac else 'no AC'}); "
          f"{stages} stage split, middle stage forward {stage_f:.2f} ms; "
          f"{PX} px encode forward {enc_f:.2f} / forward + backward {enc_fb:.2f} ms; image = {TOKENS} tokens")
    print("| seq | cap | mean images per micro-batch | vision tokens share | vision share of step compute | text : vision "
          "| cost ratio |")
    print("|---:|---:|---:|---:|---:|---|---:|")
    for seq in (4096, 6144, 8192):
        fit = max(0, (seq - 96) // (TOKENS + 2))
        for cap in range(1, min(3, fit) + 1):
            images = sum(min(n, cap) * c for n, c in enumerate(counts))
            mean, per_carrier = images / total, images / (total - counts[0])
            vision = mean * (enc_f + enc_fb)
            share = vision / (vision + model_2048 * seq / 2048)
            ratio = per_carrier * enc_f / (stage_f * seq / 2048)
            print(f"| {seq} | {cap} | {mean:.2f} | {mean * TOKENS / seq:.0%} | {share:.1%} | "
                  f"{100 - round(share * 100)} : {round(share * 100)} | {ratio:.2f} |")


if __name__ == "__main__":
    main()
