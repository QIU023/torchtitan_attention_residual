"""POPE F1 — yes/no hallucination probe over 3 splits × 3K samples = 9K total.

Each sample is a yes/no question about an object's presence in a COCO val2014
image. The prediction is reduced to {yes, no} via a simple keyword check on
the model's first-line output; F1 / Accuracy / Precision / Recall are
reported overall and per-split.

Time budget: 9K samples × ~0.8s/sample = ~2 hours single-GPU; with 8 GPUs
round-robin shard ≈ 15-20 minutes. Generation cap = 12 tokens.
"""
from __future__ import annotations

import argparse
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


POPE_DIR = Path(os.environ.get("POPE_DIR", "/workspace/.hf_home/eval_data/pope"))
IMAGE_DIR = POPE_DIR / "val2014"
SPLITS = ("random", "popular", "adversarial")


def _load_records(limit: int | None = None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for split in SPLITS:
        path = POPE_DIR / f"coco_pope_{split}.json"
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                out.append({
                    "id": f"{split}:{rec['question_id']}",
                    "split": split,
                    "image": rec["image"],
                    "question": rec["text"],
                    "gt": rec["label"].strip().lower(),  # "yes" or "no"
                })
                if limit and len(out) >= limit:
                    return out
    return out


def _image_loader(rec: dict[str, Any]) -> Image.Image:
    p = IMAGE_DIR / rec["image"]
    return Image.open(p).convert("RGB")


def _prompt_builder(rec: dict[str, Any]) -> str:
    # POPE official prompt convention: ask the yes/no question directly,
    # nudge the model to answer with a single word.
    return rec["question"] + "\nAnswer the question using a single word: yes or no."


def _parse_yes_no(text: str) -> str:
    """Return 'yes' / 'no' / 'unknown' based on first-line tokens.

    LLaVA eval convention: lowercase, strip punctuation, look for exact
    'yes' / 'no' tokens. If both / neither found → 'unknown'.
    """
    if not text:
        return "unknown"
    s = text.lower().split("\n", 1)[0]
    s = re.sub(r"[^a-z\s]", " ", s)
    toks = s.split()
    has_yes = "yes" in toks
    has_no = "no" in toks
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    # If both or neither: tie-break by first occurrence; else unknown.
    for t in toks:
        if t == "yes":
            return "yes"
        if t == "no":
            return "no"
    return "unknown"


def _score(preds: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute overall + per-split F1/precision/recall/accuracy."""
    # Group by split (split is encoded in id prefix).
    by_split: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    by_split["__all__"] = []
    for p in preds:
        pred = _parse_yes_no(p["pred"])
        gt = p["gt"]
        split = p["id"].split(":", 1)[0]
        rec = {"pred": pred, "gt": gt}
        by_split["__all__"].append(rec)
        if split in by_split:
            by_split[split].append(rec)

    def metrics(rows):
        if not rows:
            return {"n": 0}
        # positive class = "yes"
        tp = fp = fn = tn = 0
        correct = 0
        unknown = 0
        for r in rows:
            if r["pred"] == "unknown":
                unknown += 1
                # treat unknown as incorrect for accuracy; for F1 treat as
                # predicting the opposite of gt (worst case) — but standard
                # LLaVA POPE eval treats unknown as 'no'.
                pred_for_f1 = "no"
            else:
                pred_for_f1 = r["pred"]
            if pred_for_f1 == r["gt"]:
                correct += 1
            if pred_for_f1 == "yes" and r["gt"] == "yes":
                tp += 1
            elif pred_for_f1 == "yes" and r["gt"] == "no":
                fp += 1
            elif pred_for_f1 == "no" and r["gt"] == "yes":
                fn += 1
            elif pred_for_f1 == "no" and r["gt"] == "no":
                tn += 1
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        acc = correct / len(rows)
        yes_ratio = sum(1 for r in rows if r["pred"] == "yes") / len(rows)
        return {
            "n": len(rows),
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "yes_ratio_in_preds": round(yes_ratio, 4),
            "unknown_count": unknown,
        }

    return {
        "overall": metrics(by_split["__all__"]),
        "per_split": {s: metrics(by_split[s]) for s in SPLITS},
        "primary_metric": "f1",
        "primary_score": metrics(by_split["__all__"]).get("f1", 0.0),
    }


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--eval.output-dir", dest="output_dir", required=True)
    ap.add_argument("--eval.limit", dest="limit", type=int, default=0,
                    help="Cap total records (0 = all 9K). For smoke tests.")
    ap.add_argument("--eval.max-new-tokens", dest="max_new_tokens", type=int, default=12)
    args, remaining = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    records = _load_records(limit=args.limit or None)
    run_benchmark(
        name="pope",
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
