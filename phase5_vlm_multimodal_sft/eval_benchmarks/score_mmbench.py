"""MMBench-en dev split — 4,329 4-way multiple-choice questions.

Each row has an inline image (bytes), question text, optional hint, and
4 options A/B/C/D. Ground truth is the option letter. Accuracy = #correct / N.

Scoring: parse the model's free-form output for the first option letter
(A/B/C/D) — robust to "The answer is C", "C.", "(C)", etc.

Time budget: 4.3K × ~0.6s = ~45min single GPU; ~5-10min 8 GPUs.
"""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from pathlib import Path
from typing import Any

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
from phase5_vlm_multimodal_sft.eval_benchmarks.eval_common import run_benchmark  # noqa: E402


MMB_DIR = Path(os.environ.get("MMB_DIR", "/workspace/.hf_home/eval_data/mmbench/en"))


def _load_records(limit: int | None = None) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq
    p = MMB_DIR / "dev-00000-of-00001.parquet"
    df = pq.read_table(p).to_pandas()
    if limit:
        df = df.head(limit)
    records: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        img = row["image"]
        img_bytes = img["bytes"] if isinstance(img, dict) else img
        rec = {
            "id": int(row["index"]),
            "question": str(row["question"]),
            "hint": str(row["hint"]) if isinstance(row["hint"], str) else "",
            "A": str(row["A"]) if isinstance(row["A"], str) else "",
            "B": str(row["B"]) if isinstance(row["B"], str) else "",
            "C": str(row["C"]) if isinstance(row["C"], str) else "",
            "D": str(row["D"]) if isinstance(row["D"], str) else "",
            "gt": str(row["answer"]).strip().upper()[:1],  # 'A'/'B'/'C'/'D'
            "category": str(row.get("category") or ""),
            "_img_bytes": img_bytes,
        }
        records.append(rec)
    return records


def _image_loader(rec: dict[str, Any]) -> Image.Image:
    return Image.open(io.BytesIO(rec["_img_bytes"])).convert("RGB")


def _prompt_builder(rec: dict[str, Any]) -> str:
    parts: list[str] = []
    if rec["hint"] and rec["hint"].lower() not in ("nan", "none"):
        parts.append(f"Hint: {rec['hint']}")
    parts.append(rec["question"])
    opts: list[str] = []
    for letter in ("A", "B", "C", "D"):
        v = rec.get(letter, "")
        if v and v.lower() not in ("nan", "none"):
            opts.append(f"{letter}. {v}")
    parts.append("\n".join(opts))
    parts.append(
        "Answer with the option's letter from the given choices directly."
    )
    return "\n".join(parts)


_LETTER_RE = re.compile(r"\b([A-D])\b")


def _parse_letter(text: str) -> str:
    """Extract first A/B/C/D from output."""
    s = (text or "").strip()
    # First try the very first character if it's a letter.
    if s and s[0].upper() in "ABCD":
        return s[0].upper()
    m = _LETTER_RE.search(s.upper())
    if m:
        return m.group(1)
    return "?"


def _score(preds: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(preds)
    if n == 0:
        return {"accuracy": 0.0, "primary_metric": "accuracy", "primary_score": 0.0}
    correct = 0
    parsed = 0
    for p in preds:
        pred = _parse_letter(p["pred"])
        if pred in "ABCD":
            parsed += 1
        if pred == p["gt"]:
            correct += 1
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
    # Strip large image bytes from records before printing; keep for loader.
    run_benchmark(
        name="mmbench_en_dev",
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
