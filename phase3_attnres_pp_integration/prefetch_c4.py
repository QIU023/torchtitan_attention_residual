#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-download C4-en shards into the HuggingFace cache.

Runtime C4 streaming hit an ``httpx.ClientClosedError`` mid-run on our
Phase 2 N=12 ablation. On an 8-GPU rental run that error would cost
hundreds of dollars. This script downloads a fixed set of shards up
front so ``datasets.load_dataset("allenai/c4", ..., streaming=True)``
can serve them from local disk on subsequent runs.

Usage (from the rental box before launching 8-GPU training):

    # Default: first 150 shards (~45 GB), enough for 20B tokens
    python phase3_attnres_pp_integration/prefetch_c4.py

    # Smaller: 10 shards (~3 GB) for Phase 3 smoke runs
    python phase3_attnres_pp_integration/prefetch_c4.py --n_shards 10

    # Custom cache path
    HF_HOME=/mnt/ssd/hfcache python phase3_attnres_pp_integration/prefetch_c4.py --n_shards 150

Notes:
- Downloads compressed .json.gz files. runtime streaming decompresses
  on the fly; we do NOT pre-tokenize (tokenized int32 storage is
  actually larger than gzipped text).
- Verifies by loading one sample from each shard; raises if any shard
  is corrupt or incomplete.
- Resumable: if HF cache already has a shard, hf_hub_download is a no-op.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download


REPO_ID = "allenai/c4"
# Commit hash pinned in torchtitan's HuggingFaceTextDataset config for
# reproducibility. Matches the revision the streaming loader already
# resolves to. If you use a different revision, update this constant.
REVISION = "1588ec454efa1a09f29cd18ddd04fe05fc8653a2"


def _shard_filename(i: int) -> str:
    return f"en/c4-train.{i:05d}-of-01024.json.gz"


def _download_one(shard_idx: int, cache_dir: str | None) -> tuple[int, str, int]:
    path = hf_hub_download(
        repo_id=REPO_ID,
        filename=_shard_filename(shard_idx),
        repo_type="dataset",
        revision=REVISION,
        cache_dir=cache_dir,
    )
    size = os.path.getsize(path)
    return shard_idx, path, size


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--n_shards", type=int, default=150,
        help="Number of shards to prefetch starting from shard 0. "
             "Rule of thumb: 150M tokens per shard. Default 150 = 22B tokens.",
    )
    ap.add_argument(
        "--start_shard", type=int, default=0,
        help="First shard index to download (useful for resuming a "
             "partial prefetch or sharding the prefetch across machines).",
    )
    ap.add_argument(
        "--cache_dir", default=None,
        help="Override HF cache directory. If unset, uses $HF_HOME or "
             "~/.cache/huggingface.",
    )
    ap.add_argument(
        "--parallel", type=int, default=8,
        help="Parallel download workers. HF Hub tolerates 8-16 parallel.",
    )
    args = ap.parse_args()

    end_shard = args.start_shard + args.n_shards
    shards = list(range(args.start_shard, end_shard))
    print(f"[prefetch_c4] downloading {len(shards)} shards "
          f"({shards[0]} .. {shards[-1]}) from {REPO_ID}@{REVISION[:8]} "
          f"into cache={args.cache_dir or os.environ.get('HF_HOME', '~/.cache/huggingface')}")

    total_bytes = 0
    start = time.time()
    done = 0
    failed: list[int] = []
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(_download_one, i, args.cache_dir): i for i in shards}
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                idx, path, size = fut.result()
                total_bytes += size
                done += 1
                if done % 10 == 0 or done == len(shards):
                    elapsed = time.time() - start
                    gb = total_bytes / 1e9
                    mbps = (total_bytes * 8 / 1e6) / max(elapsed, 1e-3)
                    print(
                        f"[prefetch_c4] {done}/{len(shards)} shards, "
                        f"{gb:.2f} GB, elapsed {elapsed:.0f}s, "
                        f"avg {mbps:.1f} Mbps"
                    )
            except Exception as e:
                failed.append(i)
                print(f"[prefetch_c4] shard {i} FAILED: {e}")

    elapsed = time.time() - start
    print(f"[prefetch_c4] done: {done}/{len(shards)} shards, "
          f"{total_bytes/1e9:.2f} GB in {elapsed:.0f}s "
          f"({(total_bytes/1e9)/max(elapsed/3600, 1e-3):.1f} GB/h)")
    if failed:
        print(f"[prefetch_c4] FAILED shards: {failed}")
        return 1

    print("[prefetch_c4] OK. Verify downstream with:")
    print('    python -c "from datasets import load_dataset; '
          'ds = load_dataset(\\"allenai/c4\\", \\"en\\", split=\\"train\\", streaming=True); '
          'print(next(iter(ds)))"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
