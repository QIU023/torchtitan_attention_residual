"""MMMU validation split — ~900 multimodal multiple-choice questions.

Each record may contain up to 7 images (<image 1>..<image 7>). Our model
is single-image (1 SigLIP slot × 196 tokens). To keep this offline-only
script compatible without architectural changes, we use ONLY image_1 and
replace all <image N> placeholders in the question with a generic "[image]"
tag. Multi-image MMMU questions will likely underperform; that's an
inherent limitation of our 1-image VLM design and worth reporting.

Time budget: 900 × ~0.7s = ~10min single GPU; ~2min on 8 GPUs.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import string
import sys
from pathlib import Path
from typing import Any

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from phase5_vlm_multimodal_sft.eval_benchmarks.eval_common import run_benchmark  # noqa: E402


MMMU_DIR = Path(os.environ.get("MMMU_DIR", "/workspace/.hf_home/eval_data/mmmu/data"))


def _load_records(limit: int | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    p = MMMU_DIR / "validation-00000-of-00001.parquet"
    df = pq.read_table(p).to_pandas()
    if limit:
        df = df.head(limit)
    records: list[dict[str, Any]] = []
    skipped_no_img = 0
    skipped_not_mcq = 0
    for _, row in df.iterrows():
        img = row["image_1"]
        if img is None:
            skipped_no_img += 1
            continue
        img_bytes = img["bytes"] if isinstance(img, dict) else img
        raw_opts = row["options"]
        # MMMU stores options as a STRING that looks like a Python list
        # (e.g. "['$6', '$7', '$8', '$9']") rather than an actual list.
        # Parse defensively: if it's already a list, use it directly;
        # if it's a string, try literal_eval; otherwise fall back to a
        # single-option list and log a warning.
        if raw_opts is None:
            options = []
        elif isinstance(raw_opts, str):
            try:
                import ast
                options = ast.literal_eval(raw_opts)
                if not isinstance(options, (list, tuple)):
                    options = [str(options)]
            except Exception:
                options = [raw_opts]
        else:
            options = list(raw_opts)
        if not options:
            skipped_not_mcq += 1
            continue
        gt = str(row["answer"]).strip().upper()[:1]
        records.append({
            "id": str(row["id"]),
            "question": str(row["question"]),
            "options": [str(o) for o in options],
            "gt": gt,
            "_img_bytes": img_bytes,
            "subfield": str(row.get("subfield") or ""),
        })
    if skipped_no_img or skipped_not_mcq:
        print(
            f"mmmu: skipped {skipped_no_img} text-only, "
            f"{skipped_not_mcq} non-MCQ", file=sys.stderr,
        )
    return records


def _image_loader(rec: dict[str, Any]) -> Image.Image:
    return Image.open(io.BytesIO(rec["_img_bytes"])).convert("RGB")


_IMG_TAG_RE = re.compile(r"<image\s*\d+>", re.IGNORECASE)


def _prompt_builder(rec: dict[str, Any]) -> str:
    q = _IMG_TAG_RE.sub("[image]", rec["question"])
    opts = []
    for i, opt in enumerate(rec["options"]):
        letter = string.ascii_uppercase[i]
        opts.append(f"{letter}. {opt}")
    return (
        q + "\n" + "\n".join(opts)
        + "\nAnswer with the option's letter from the given choices directly."
    )


_LETTER_RE = re.compile(r"\b([A-J])\b")


def _parse_letter(text: str) -> str:
    s = (text or "").strip()
    if s and s[0].upper() in string.ascii_uppercase[:10]:
        return s[0].upper()
    m = _LETTER_RE.search(s.upper())
    if m:
        return m.group(1)
    return "?"


def _score(preds: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(preds)
    if n == 0:
        return {"accuracy": 0.0, "primary_metric": "accuracy", "primary_score": 0.0}
    correct = sum(1 for p in preds if _parse_letter(p["pred"]) == p["gt"])
    parsed = sum(1 for p in preds if _parse_letter(p["pred"]) in string.ascii_uppercase[:10])
    acc = correct / n
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "parse_rate": round(parsed / n, 4),
        "primary_metric": "accuracy",
        "primary_score": round(acc, 4),
    }


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--eval.output-dir", dest="output_dir", required=True)
    ap.add_argument("--eval.limit", dest="limit", type=int, default=0)
    ap.add_argument("--eval.max-new-tokens", dest="max_new_tokens", type=int, default=8)
    args, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    records = _load_records(limit=args.limit or None)
    run_benchmark(
        name="mmmu_val",
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
