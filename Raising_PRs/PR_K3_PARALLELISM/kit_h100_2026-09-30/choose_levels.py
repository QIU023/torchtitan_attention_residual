"""Vision share of a DEP step for candidate (seq, images-per-sample cap) settings, from measured times.

Inputs: microbench_ratio.py's output (text stage forward and forward + backward at seq 2048, encode forward and
forward + backward per image) and the dataset MANIFEST (n_images_counts). A micro-batch is one sample padded to seq
(the multimodal loader does not pack), so the text side costs seq tokens whatever the image count; the tower costs
its forward, then recompute plus backward (DEP's), per image. Text time scales linearly with seq from 2048 (the MLA
layers' attention grows faster, so long seqs slightly overstate the vision share). Stages per micro-batch: 8
(pp2 x vpp4); the share is per step, all ranks together.
"""

import json
import re
import sys

PX, TOKENS = 1024, (1008 // 14 // 2) ** 2


def main():
    text = open(sys.argv[1]).read()
    m = re.search(r"text stage forward ([\d.]+) ms, forward \+ backward ([\d.]+) ms", text)
    stage_f, stage_fb = float(m.group(1)), float(m.group(2))
    row = re.search(rf"\| {PX} \| \d+ \| ([\d.]+) \| ([\d.]+) \|", text)
    enc_f, enc_fb = float(row.group(1)), float(row.group(2))
    counts = json.load(open(sys.argv[2]))["n_images_counts"]
    total = sum(counts)
    print(f"measured at seq 2048: stage fwd {stage_f:.2f} / fwd+bwd {stage_fb:.2f} ms; "
          f"{PX} px encode fwd {enc_f:.2f} / fwd+bwd {enc_fb:.2f} ms; image = {TOKENS} tokens")
    print("| seq | cap | mean images per micro-batch | vision tokens share | vision share of step compute | text : vision |")
    print("|---:|---:|---:|---:|---:|---|")
    for seq in (4096, 8192, 16384):
        fit = max(0, (seq - 96) // (TOKENS + 2))
        for cap in range(1, min(4, fit) + 1):
            mean = sum(min(n, cap) * c for n, c in enumerate(counts)) / total
            vision = mean * (enc_f + enc_fb)
            text_step = 8 * stage_fb * seq / 2048
            share = vision / (vision + text_step)
            print(f"| {seq} | {cap} | {mean:.2f} | {mean * TOKENS / seq:.0%} | {share:.1%} | "
                  f"{100 - round(share * 100)} : {round(share * 100)} |")


if __name__ == "__main__":
    main()
