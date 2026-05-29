"""Fix the OCR-VQA gap in download_instruct_665k.sh.

The original snapshot_download call uses
  allow_patterns=['images/*', 'images.zip']
but the howard-hou/OCR-VQA HF repo ships its images embedded in
``data/train-*.parquet`` columns (16 shards train + 2 shards test).
With those patterns, no files match and the download silently no-ops.

This script:
  1. Pulls all data/*.parquet shards (~9-10 GB)
  2. Iterates rows, decodes the embedded image bytes
  3. Saves each as <image_id>.jpg at the path mix665k expects
     (DEST/images/ocr_vqa/images/<id>.jpg)
  4. Cleans up the parquet shards after extraction (~9G recovered)

Idempotent: skips images already on disk; resumable on the parquet shards
via snapshot_download's own cache.
"""
from __future__ import annotations

import io
import os
import shutil
import sys
import time

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

DEST = os.environ.get("DEST", "/workspace/.hf_home/LLaVA-Instruct")
OUT_DIR = os.path.join(DEST, "images", "ocr_vqa", "images")
PARQUET_DIR = os.path.join(DEST, "_ocrvqa_parquet_tmp")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PARQUET_DIR, exist_ok=True)

print(f"[ocrvqa] target: {OUT_DIR}", flush=True)

# ---- 1. Pull parquet shards (resumable) ----
from huggingface_hub import snapshot_download

print(f"[ocrvqa] snapshot_download howard-hou/OCR-VQA data/*.parquet", flush=True)
t0 = time.time()
local = snapshot_download(
    repo_id="howard-hou/OCR-VQA",
    repo_type="dataset",
    local_dir=PARQUET_DIR,
    allow_patterns=["data/*.parquet"],
)
print(f"[ocrvqa] downloaded shards to {local} ({time.time()-t0:.1f}s)", flush=True)

# ---- 2. Inspect schema (first row of first shard) ----
import glob

import pyarrow.parquet as pq

shards = sorted(glob.glob(os.path.join(local, "data", "*.parquet")))
if not shards:
    print(f"FATAL: no parquet shards under {local}/data/")
    sys.exit(1)
print(f"[ocrvqa] {len(shards)} shards", flush=True)

# Schema probe
table0 = pq.read_table(shards[0], columns=None)
print(f"[ocrvqa] schema: {table0.schema.names}", flush=True)
sample = table0.to_pandas().iloc[0]
for col in table0.schema.names:
    v = sample[col]
    if isinstance(v, (bytes, bytearray)):
        print(f"   {col}: <bytes len={len(v)}>")
    elif isinstance(v, dict):
        print(f"   {col}: dict keys={list(v.keys())}")
    else:
        print(f"   {col}: {type(v).__name__} = {str(v)[:80]}")

# ---- 3. Extract images shard by shard ----
def find_id_col(names: list[str]) -> str:
    for cand in ("image_id", "question_id", "id"):
        if cand in names:
            return cand
    return names[0]


def get_image_bytes(row) -> bytes | None:
    img = row.get("image")
    if isinstance(img, dict):
        # HF datasets style: {'bytes': b'...', 'path': '...'}
        return img.get("bytes")
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)
    return None


id_col = find_id_col(table0.schema.names)
print(f"[ocrvqa] id column = '{id_col}'", flush=True)
del table0

n_extracted = 0
n_skipped = 0
n_errors = 0
t0 = time.time()
for i, sh in enumerate(shards):
    sh_t0 = time.time()
    df = pq.read_table(sh).to_pandas()
    for _, row in df.iterrows():
        img_id = row[id_col]
        img_bytes = get_image_bytes(row)
        if img_bytes is None:
            n_errors += 1
            continue
        out_path = os.path.join(OUT_DIR, f"{img_id}.jpg")
        if os.path.exists(out_path):
            n_skipped += 1
            continue
        try:
            with open(out_path, "wb") as f:
                f.write(img_bytes)
            n_extracted += 1
        except Exception as e:
            n_errors += 1
            if n_errors < 5:
                print(f"[ocrvqa] write fail {img_id}: {e}", flush=True)
    print(f"[ocrvqa] shard {i+1}/{len(shards)} done in {time.time()-sh_t0:.1f}s "
          f"(extracted={n_extracted} skipped={n_skipped} errors={n_errors})",
          flush=True)
    del df

print(f"\n[ocrvqa] TOTAL: extracted={n_extracted} skipped={n_skipped} errors={n_errors} "
      f"in {time.time()-t0:.1f}s", flush=True)

# ---- 4. Cleanup parquet shards ----
print(f"[ocrvqa] removing parquet shards to recover disk", flush=True)
shutil.rmtree(PARQUET_DIR, ignore_errors=True)

print(f"[ocrvqa] DONE — images in {OUT_DIR}", flush=True)
