"""ScienceQA-IMG test split — multimodal multiple-choice subset.

We restrict to records with an attached image (the IMG-only variant, ~2K).
Each record has free-form `choices` (list of strings) and an integer
`answer` (0-indexed); we map the prediction letter back to its position.

Time budget: ~2K × ~0.6s = ~20min single GPU; ~3-5min 8 GPUs.
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


SQA_DIR = Path(os.environ.get("SQA_DIR", "/workspace/.hf_home/eval_data/scienceqa/data"))


def _load_records(limit: int | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    p = SQA_DIR / "test-00000-of-00001-f0e719df791966ff.parquet"
    df = pq.read_table(p).to_pandas()
    df = df[df["image"].notnull()].reset_index(drop=True)
    if limit:
        df = df.head(limit)
    records: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        img = row["image"]
        img_bytes = img["bytes"] if isinstance(img, dict) else img
        choices = list(row["choices"]) if row["choices"] is not None else []
        gt_idx = int(row["answer"]) if row["answer"] is not None else -1
        if not choices or gt_idx < 0 or gt_idx >= len(choices):
            continue
        gt_letter = string.ascii_uppercase[gt_idx]
        records.append({
            "id": f"sqa:{i}",
            "question": str(row["question"]),
            "hint": str(row["hint"]) if isinstance(row["hint"], str) else "",
            "choices": choices,
            "gt": gt_letter,
            "_img_bytes": img_bytes,
            "topic": str(row.get("topic") or ""),
            "subject": str(row.get("subject") or ""),
        })
    return records


def _image_loader(rec: dict[str, Any]) -> Image.Image:
    return Image.open(io.BytesIO(rec["_img_bytes"])).convert("RGB")


def _prompt_builder(rec: dict[str, Any]) -> str:
    parts: list[str] = []
    if rec["hint"] and rec["hint"].strip():
        parts.append(f"Context: {rec['hint']}")
    parts.append(rec["question"])
    opts: list[str] = []
    for i, c in enumerate(rec["choices"]):
        letter = string.ascii_uppercase[i]
        opts.append(f"{letter}. {c}")
    parts.append("\n".join(opts))
    parts.append("Answer with the option's letter from the given choices directly.")
    return "\n".join(parts)


_LETTER_RE = re.compile(r"\b([A-H])\b")


def _parse_letter(text: str) -> str:
    s = (text or "").strip()
    if s and s[0].upper() in string.ascii_uppercase[:8]:
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
    parsed = sum(
        1 for p in preds if _parse_letter(p["pred"]) in string.ascii_uppercase
    )
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
        name="scienceqa_img",
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
