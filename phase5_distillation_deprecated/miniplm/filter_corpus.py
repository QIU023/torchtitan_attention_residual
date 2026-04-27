#!/usr/bin/env python3
"""Filter scored c4 chunks down to a training corpus by score.

Reads all scored_*.jsonl shards from --in-dir, ranks by score
(teacher_logp - reference_logp), keeps the top `--keep-ratio` fraction,
and writes a single filtered.jsonl ready for continued pretraining.

Default keep-ratio = 0.5 (top 50% of chunks). Per MiniPLM the optimal
fraction depends on dataset; 0.5 is a safe starting point and matches
their "diff_samp-r0.5" preprocessed corpus name.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True,
                   help="Directory containing scored_*.jsonl files")
    p.add_argument("--out", required=True, help="Output filtered.jsonl path")
    p.add_argument("--keep-ratio", type=float, default=0.5)
    p.add_argument("--mode", choices=["top", "weighted"], default="top",
                   help="'top' = keep top-K by score; "
                        "'weighted' = sample with prob proportional to softmax(score)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Softmax temperature for weighted mode")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect (score, line_offset, file_path) tuples — keep input_ids on disk
    # to avoid loading entire corpus into RAM.
    shards = sorted(in_dir.glob("scored_*.jsonl"))
    if not shards:
        raise SystemExit(f"No scored_*.jsonl in {in_dir}")
    print(f"[filter] reading {len(shards)} shards from {in_dir}")

    # Index: per shard, list of (offset, score)
    index: list[tuple[Path, int, float]] = []
    for shard in shards:
        with shard.open("r") as f:
            offset = 0
            for line in f:
                rec = json.loads(line)
                index.append((shard, offset, rec["score"]))
                offset += len(line.encode("utf-8"))
        print(f"[filter] {shard.name}: {len(index)} cumulative records")

    print(f"[filter] total scored chunks: {len(index)}")

    # Select chunks
    if args.mode == "top":
        index.sort(key=lambda x: x[2], reverse=True)
        n_keep = int(len(index) * args.keep_ratio)
        kept = index[:n_keep]
        print(f"[filter] mode=top keep_ratio={args.keep_ratio} "
              f"-> keeping {n_keep} chunks "
              f"(score range: {kept[-1][2]:.3f} to {kept[0][2]:.3f})")
    else:
        import math
        import random
        random.seed(args.seed)
        T = args.temperature
        scores = [x[2] / T for x in index]
        # softmax in log-space for stability
        m = max(scores)
        denom = sum(math.exp(s - m) for s in scores)
        probs = [math.exp(s - m) / denom for s in scores]
        n_keep = int(len(index) * args.keep_ratio)
        kept_idx = random.choices(range(len(index)), weights=probs, k=n_keep)
        kept = [index[i] for i in kept_idx]
        print(f"[filter] mode=weighted keep_ratio={args.keep_ratio} "
              f"T={T} -> sampled {n_keep} chunks")

    # Write filtered corpus — re-read each chunk by file offset
    print(f"[filter] writing to {out_path} ...")
    n_written = 0
    with out_path.open("w") as out_f:
        for shard, offset, _score in kept:
            with shard.open("r") as f:
                f.seek(offset)
                line = f.readline()
                # Re-emit only what training needs: input_ids
                rec = json.loads(line)
                out_f.write(json.dumps({"input_ids": rec["input_ids"]}) + "\n")
            n_written += 1
            if n_written % 50_000 == 0:
                print(f"[filter] wrote {n_written}/{len(kept)}")
    print(f"[filter] done. {n_written} chunks in {out_path}")


if __name__ == "__main__":
    main()
