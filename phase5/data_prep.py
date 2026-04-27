#!/usr/bin/env python3
"""Download LLaVA-Pretrain-558K + SigLIP vision tower.

LLaVA-Pretrain has two parts:
1. `blip_laion_cc_sbu_558k.json` — caption pairs (image filename + text)
2. `images.zip` — the actual JPEG images

We download both, unpack images, save the json as-is. A separate
multimodal_dataset.py reads them at training time.

Sigil vision tower is downloaded via transformers' from_pretrained
into the standard HF cache layout under HF_HOME (= /root/hf_cache).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="/root/hf_cache/LLaVA-Pretrain")
    p.add_argument("--vision-model", default="google/siglip-base-patch16-224")
    p.add_argument("--cache-dir", default="/root/hf_cache")
    return p.parse_args()


def download_llava_pretrain(out_dir: Path):
    """Download LLaVA-Pretrain dataset (json + images.zip)."""
    from huggingface_hub import snapshot_download
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] downloading liuhaotian/LLaVA-Pretrain → {out_dir}")
    t0 = time.perf_counter()
    snapshot_download(
        repo_id="liuhaotian/LLaVA-Pretrain",
        repo_type="dataset",
        local_dir=str(out_dir),
        max_workers=8,
    )
    dt = time.perf_counter() - t0
    print(f"[data] download finished in {dt/60:.1f} min")

    # Inspect
    files = list(out_dir.iterdir())
    print(f"[data] files in {out_dir}:")
    for f in files:
        size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else None
        suffix = f"  ({size_mb:.1f} MB)" if size_mb else ""
        print(f"  {f.name}{suffix}")

    # Unzip images.zip if present
    images_zip = out_dir / "images.zip"
    images_dir = out_dir / "images"
    if images_zip.exists() and not images_dir.exists():
        print(f"[data] unzipping {images_zip} → {images_dir}")
        t0 = time.perf_counter()
        import zipfile
        with zipfile.ZipFile(images_zip, "r") as z:
            z.extractall(str(out_dir))
        dt = time.perf_counter() - t0
        print(f"[data] unzip finished in {dt/60:.1f} min")
    elif images_dir.exists():
        print(f"[data] images dir already exists, skipping unzip")

    # Verify json
    json_files = sorted(out_dir.glob("*.json"))
    if not json_files:
        raise RuntimeError(f"No JSON file found in {out_dir}")
    main_json = json_files[0]
    print(f"[data] caption json: {main_json}")
    with main_json.open("r") as f:
        records = json.load(f)
    print(f"[data] caption records: {len(records):,}")
    if records:
        sample = records[0]
        print(f"[data] sample record keys: {list(sample.keys())}")
        print(f"[data] sample: {json.dumps(sample, indent=2)[:500]}")


def download_vision_model(model_name: str, cache_dir: str):
    """Trigger HF cache populate for the SigLIP model."""
    from transformers import AutoModel, AutoProcessor
    print(f"[data] downloading vision model {model_name} → {cache_dir}")
    t0 = time.perf_counter()
    AutoModel.from_pretrained(model_name, cache_dir=cache_dir)
    AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
    dt = time.perf_counter() - t0
    print(f"[data] vision model finished in {dt/60:.1f} min")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    download_llava_pretrain(out_dir)
    download_vision_model(args.vision_model, args.cache_dir)

    print("\n[data] DONE.")
    print(f"  caption json:    {out_dir}/<*.json>")
    print(f"  images dir:      {out_dir}/images/")
    print(f"  vision model:    HF cache at {args.cache_dir}/models--google--*")


if __name__ == "__main__":
    main()
