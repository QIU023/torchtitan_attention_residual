"""Post-torchrun aggregator: collect per-rank preds_rank*.jsonl + score.

This runs as a single CPU process AFTER ``torchrun`` exits (regardless of
exit code), so a crash on any rank still produces a partial result.json.
Each benchmark exports its scorer + total record count via the
``BENCH_SCORERS`` registry below; new benchmarks just need to register
their (scorer, total_records_fn) here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))

from phase5_vlm_multimodal_sft.eval_benchmarks.eval_common import (  # noqa: E402
    score_benchmark_from_files,
)


def _pope_total(limit: int) -> int:
    from phase5_vlm_multimodal_sft.eval_benchmarks.score_pope import _load_records
    return len(_load_records(limit=limit or None))


def _gqa_total(limit: int) -> int:
    from phase5_vlm_multimodal_sft.eval_benchmarks.score_gqa import _load_records
    records, _ = _load_records(limit=limit or None)
    return len(records)


def _mmbench_total(limit: int) -> int:
    from phase5_vlm_multimodal_sft.eval_benchmarks.score_mmbench import _load_records
    return len(_load_records(limit=limit or None))


def _scienceqa_total(limit: int) -> int:
    from phase5_vlm_multimodal_sft.eval_benchmarks.score_scienceqa import _load_records
    return len(_load_records(limit=limit or None))


def _mmmu_total(limit: int) -> int:
    from phase5_vlm_multimodal_sft.eval_benchmarks.score_mmmu import _load_records
    return len(_load_records(limit=limit or None))


REGISTRY: dict[str, tuple[str, callable, callable]] = {
    "pope": ("pope", lambda: __import__(
        "phase5_vlm_multimodal_sft.eval_benchmarks.score_pope",
        fromlist=["_score"]
    )._score, _pope_total),
    "gqa": ("gqa", lambda: __import__(
        "phase5_vlm_multimodal_sft.eval_benchmarks.score_gqa",
        fromlist=["_score"]
    )._score, _gqa_total),
    "mmbench": ("mmbench_en_dev", lambda: __import__(
        "phase5_vlm_multimodal_sft.eval_benchmarks.score_mmbench",
        fromlist=["_score"]
    )._score, _mmbench_total),
    "scienceqa": ("scienceqa_img", lambda: __import__(
        "phase5_vlm_multimodal_sft.eval_benchmarks.score_scienceqa",
        fromlist=["_score"]
    )._score, _scienceqa_total),
    "mmmu": ("mmmu_val", lambda: __import__(
        "phase5_vlm_multimodal_sft.eval_benchmarks.score_mmmu",
        fromlist=["_score"]
    )._score, _mmmu_total),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=list(REGISTRY.keys()))
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    bench_label, scorer_loader, total_fn = REGISTRY[args.bench]
    scorer = scorer_loader()
    n_total = total_fn(args.limit)
    result = score_benchmark_from_files(
        name=bench_label,
        output_dir=args.output_dir,
        n_total=n_total,
        scorer=scorer,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
