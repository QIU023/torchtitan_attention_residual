"""Aggregate per-benchmark result.json into a single REPORT.md table.

Reads each ``<RUN_DIR>/<bench>/result.json`` and emits a markdown table
with: benchmark, n_scored, primary metric, elapsed_sec, status. Also
includes per-split breakdown for POPE.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


PAPER_REFERENCE = {
    "pope":            ("F1",       85.9, "LLaVA-1.5-7B paper"),
    "gqa":             ("acc",      62.0, "LLaVA-1.5-7B paper"),
    "mmbench_en_dev":  ("acc",      64.3, "LLaVA-1.5-7B paper"),
    "scienceqa_img":   ("acc",      66.8, "LLaVA-1.5-7B paper"),
    "mmmu_val":        ("acc",      36.4, "LLaVA-1.5-7B paper"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True,
                    help="dir with subfolders <bench>/result.json")
    ap.add_argument("--ckpt", required=True, help="ckpt path label")
    ap.add_argument("--out", required=True, help="REPORT.md output path")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    results: list[dict] = []
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        rj = sub / "result.json"
        if not rj.exists():
            continue
        with open(rj) as f:
            results.append(json.load(f))

    lines = []
    lines.append(f"# VLM Downstream Eval Report")
    lines.append("")
    lines.append(f"- **Generated**: {datetime.utcnow().isoformat()}Z")
    lines.append(f"- **Checkpoint**: `{args.ckpt}`")
    lines.append(f"- **Eval backend**: in-tree torchtitan MultimodalTrainer loader + greedy decode (no sglang, no DCP→HF)")
    lines.append(f"- **Parallelism**: 1D FSDP=8 (8× RTX 5090, 32GB)")
    lines.append(f"- **Vision tower**: google/siglip-base-patch16-224 (frozen)")
    lines.append(f"- **Tokenizer**: NousResearch/Meta-Llama-3.1-8B")
    lines.append("")
    lines.append("## Summary table")
    lines.append("")
    lines.append("| Benchmark | N scored | Status | Primary metric | Score | Paper (LLaVA-1.5-7B) | Δ vs paper | Elapsed (s) |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|")
    for r in results:
        b = r["benchmark"]
        n = r.get("n_scored", "?")
        st = r.get("status", "?")
        # primary metric / score (POPE nests under "overall")
        if b == "pope":
            ov = r.get("overall", {})
            pm = "f1"
            ps = ov.get("f1", float("nan"))
        else:
            pm = r.get("primary_metric", "?")
            ps = r.get("primary_score", float("nan"))
        ref = PAPER_REFERENCE.get(b, ("-", float("nan"), ""))
        delta = ps * 100 - ref[1] if isinstance(ps, (int, float)) and ref[1] == ref[1] else float("nan")
        elapsed = r.get("elapsed_sec", float("nan"))
        try:
            ps_str = f"{ps*100:.2f}" if isinstance(ps, (int, float)) else str(ps)
        except Exception:
            ps_str = "?"
        ref_str = f"{ref[1]:.1f}" if ref[1] == ref[1] else "-"
        delta_str = f"{delta:+.2f}" if delta == delta else "-"
        elapsed_str = f"{elapsed:.0f}" if elapsed == elapsed else "-"
        lines.append(
            f"| `{b}` | {n} | {st} | {pm} | {ps_str} | {ref_str} | {delta_str} | {elapsed_str} |"
        )
    lines.append("")
    lines.append("Scores are × 100 (i.e. percentage points). Paper references are for the 7B baseline; our model is **447M params**, so a multi-point gap is expected.")
    lines.append("")

    # POPE per-split detail
    for r in results:
        if r["benchmark"] != "pope":
            continue
        ps = r.get("per_split", {})
        if not ps:
            continue
        lines.append("## POPE per-split detail")
        lines.append("")
        lines.append("| Split | N | F1 | Acc | Precision | Recall | yes_ratio | unknown |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for split in ("random", "popular", "adversarial"):
            s = ps.get(split, {})
            if not s:
                continue
            lines.append(
                f"| {split} | {s.get('n','?')} | "
                f"{s.get('f1',0)*100:.2f} | {s.get('accuracy',0)*100:.2f} | "
                f"{s.get('precision',0)*100:.2f} | {s.get('recall',0)*100:.2f} | "
                f"{s.get('yes_ratio_in_preds',0)*100:.2f} | {s.get('unknown_count',0)} |"
            )
        lines.append("")

    # Raw per-benchmark dumps
    lines.append("## Per-benchmark raw result.json")
    lines.append("")
    for r in results:
        lines.append(f"### `{r['benchmark']}`")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(r, indent=2))
        lines.append("```")
        lines.append("")

    Path(args.out).write_text("\n".join(lines))
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
