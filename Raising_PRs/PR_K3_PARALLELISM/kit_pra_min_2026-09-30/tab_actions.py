"""Per-action block account for run_bound_5060.sh (and run_pra_h100.sh results): python tab_actions.py <dir> <layout>

For each rank and tree, over every forward and backward action of the traced step: the blocks held (store, saved
stage inputs and outputs except output 0, forward block sends still alive) against the tight bound (each block
freed at the backward of the stage that brought it) and the paper bound, in units of one [T, D] bf16 block
(PPMEM_SEQ x PPMEM_DIM x 2 bytes); the hidden-state sends still alive are listed apart, they are not blocks.
"""

import json
import os
import re
import sys

STEP = re.compile(r"step:\s+(\d+)\s+loss:\s+(-?[\d.]+)\s+grad_norm:\s+([\d.]+)")
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def losses(path):
    out = {}
    if os.path.exists(path):
        for line in ANSI.sub("", open(path, errors="replace").read()).splitlines():
            m = STEP.search(line)
            if m and float(m.group(2)) > 0:
                out.setdefault(int(m.group(1)), (m.group(2), m.group(3)))
    return out


def ranks(root, run):
    folder = os.path.join(root, run, "mem")
    out = {}
    if os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.startswith("rank") and name.endswith(".json"):
                d = json.load(open(os.path.join(folder, name)))
                out[d["rank"]] = d
    return out


def main():
    root, L = sys.argv[1], sys.argv[2]
    blk = int(os.environ.get("PPMEM_SEQ", "2048")) * int(os.environ.get("PPMEM_DIM", "2048")) * 2 / 2**30
    n = lambda gib: gib / blk  # noqa: E731
    runs = {"4656": f"4656_{L}", "pra": f"pra_{L}"}
    la, lb = losses(os.path.join(root, runs["4656"], "run.log")), losses(os.path.join(root, runs["pra"], "run.log"))
    same = sum(1 for s in la if lb.get(s) == la[s])
    print(f"## {L}: steps with equal loss and grad norm {same} / {len(la)}; one block = {blk:.4f} GiB\n")
    print("| rank | tree | actions | blocks = tight | blocks - tight, max | at action | paper - blocks, min | hidden sends alive, max | hidden sends only, max | blocks at peak action | tight there | hidden there |")
    print("|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
    for tree, run in runs.items():
        for r, d in sorted(ranks(root, run).items()):
            acts = d.get("actions", [])
            if not acts:
                continue
            diff = [round(n(a["block_gib"] - a["tight_bound_gib"]), 3) for a in acts]
            eq = sum(1 for x in diff if abs(x) < 1e-3)
            worst = max(range(len(acts)), key=lambda i: diff[i])
            pk = max(acts, key=lambda a: a["peak_gib"])
            print(
                f"| {r} | {tree} | {len(acts)} | {eq} | {diff[worst]:.2f} | {acts[worst]['action']} | "
                f"{min(n(a['paper_bound_gib'] - a['block_gib']) for a in acts):.2f} | "
                f"{max(n(a.get('hidden_sends_alive_gib', 0.0)) for a in acts):.2f} | "
                f"{max(n(a.get('hidden_sends_only_gib', 0.0)) for a in acts):.2f} | "
                f"{n(pk['block_gib']):.2f} | {n(pk['tight_bound_gib']):.2f} | {n(pk.get('hidden_sends_alive_gib', 0.0)):.2f} |"
            )
    print("\nEvery action, PR A (blocks / tight / paper / hidden sends alive, in blocks):")
    for r, d in sorted(ranks(root, runs["pra"]).items()):
        row = " ".join(
            f"{a['action']}:{n(a['block_gib']):.0f}/{n(a['tight_bound_gib']):.0f}/{n(a['paper_bound_gib']):.0f}/{n(a.get('hidden_sends_alive_gib', 0.0)):.0f}"
            for a in d.get("actions", [])
        )
        print(f"- rank {r}: {row}")


if __name__ == "__main__":
    main()
