"""GQA test-dev-balanced (12,578 short-answer questions).

Images and instructions both come as parquet files; images are stored
inline as `{'bytes': ...}` in the images parquet (398 unique images), keyed
by `id` matching `imageId` in the instructions parquet.

Scoring: official LLaVA convention — lowercase, strip punctuation, exact
match against ground truth. Reports accuracy + per-structural-type breakdown.

Time budget: 12.6K samples × ~0.6s/sample (short answers, 16 tokens max) =
~2 hours on 1 GPU; ~15-20min on 8 GPUs.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from phase5_vlm_multimodal_sft.eval_benchmarks.eval_common import run_benchmark  # noqa: E402


GQA_DIR = Path(os.environ.get(
    "GQA_DIR", "/workspace/.hf_home/eval_data/gqa",
))


def _load_image_table() -> dict[str, bytes]:
    """Map imageId → raw image bytes from the images parquet."""
    import pyarrow.parquet as pq
    p = GQA_DIR / "testdev_balanced_images" / "testdev-00000-of-00001.parquet"
    t = pq.read_table(p)
    df = t.to_pandas()
    out: dict[str, bytes] = {}
    for _, row in df.iterrows():
        img = row["image"]
        if isinstance(img, dict):
            out[str(row["id"])] = img["bytes"]
        else:
            out[str(row["id"])] = img
    return out


def _load_records(limit: int | None = None) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    import pyarrow.parquet as pq
    p = GQA_DIR / "testdev_balanced_instructions" / "testdev-00000-of-00001.parquet"
    df = pq.read_table(p).to_pandas()
    if limit:
        df = df.head(limit)
    imgs = _load_image_table()
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        types = row.get("types") or {}
        rec = {
            "id": str(row["id"]),
            "imageId": str(row["imageId"]),
            "question": str(row["question"]),
            "gt": str(row["answer"]),
            "structural": types.get("structural") if isinstance(types, dict) else None,
            "semantic": types.get("semantic") if isinstance(types, dict) else None,
        }
        if rec["imageId"] not in imgs:
            continue  # silently skip any without image
        records.append(rec)
    return records, imgs


_REC_IMG_CACHE: dict[str, bytes] = {}


def _image_loader(rec: dict[str, Any]) -> Image.Image:
    b = _REC_IMG_CACHE.get(rec["imageId"])
    if b is None:
        raise KeyError(f"image bytes missing for {rec['imageId']}")
    return Image.open(io.BytesIO(b)).convert("RGB")


def _prompt_builder(rec: dict[str, Any]) -> str:
    # LLaVA-style short-answer cue.
    return rec["question"] + "\nAnswer the question using a single word or phrase."


_ARTICLES = {"a", "an", "the"}
_PUNCT_RE = re.compile(r"[^\w\s]")


def _normalize(text: str) -> str:
    """LLaVA / GQA-style normalization: lower, strip punct, drop articles."""
    s = text.lower().strip()
    s = s.split("\n", 1)[0]
    s = _PUNCT_RE.sub(" ", s)
    toks = [t for t in s.split() if t and t not in _ARTICLES]
    return " ".join(toks)


def _score(preds: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(preds)
    if n == 0:
        return {"accuracy": 0.0, "primary_metric": "accuracy", "primary_score": 0.0}
    correct = 0
    for p in preds:
        if _normalize(p["pred"]) == _normalize(p["gt"]):
            correct += 1
    acc = correct / n
    return {
        "accuracy": round(acc, 4),
        "primary_metric": "accuracy",
        "primary_score": round(acc, 4),
        "n": n,
    }


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--eval.output-dir", dest="output_dir", required=True)
    ap.add_argument("--eval.limit", dest="limit", type=int, default=0)
    ap.add_argument("--eval.max-new-tokens", dest="max_new_tokens", type=int, default=16)
    args, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    records, imgs = _load_records(limit=args.limit or None)
    global _REC_IMG_CACHE
    _REC_IMG_CACHE = imgs

    run_benchmark(
        name="gqa",
        records=records,
        image_loader=_image_loader,
        prompt_builder=_prompt_builder,
        output_dir=args.output_dir,
        max_new_tokens=args.max_new_tokens,
        stop_on_newline=True,
        scorer=_score,
    )


if __name__ == "__main__":
    main()
