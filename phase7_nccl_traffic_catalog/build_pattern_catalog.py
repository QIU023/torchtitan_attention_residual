#!/usr/bin/env python3
"""Build phase7_nccl_traffic_catalog/pattern_catalog.md from all collected NCCL traces.

Walks ``phase5_vlm_multimodal_sft/runs/8gpu_*/tier_{a,b,c}_trace/collective_summary.csv``
and consolidates into a single markdown catalog with:

* Per-collective table (op, size_bucket, nranks) × (config, tier) →
  per-step count.
* Replay-priority table: every (config, tier) with its trace path,
  recipe, ranked by realism.
* Per-config tier comparison: shows shift in tensor sizes between
  tier C (alignment-load) and tier A (production-standardized).

Run after all alignment + tier B + tier A runs complete.

Usage:
    python phase7_nccl_traffic_catalog/build_pattern_catalog.py
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
RUNS_ROOT = WORKSPACE / "phase5" / "runs"
CATALOG = WORKSPACE / "phase7" / "pattern_catalog.md"

# Order matters: lower index = more realistic
TIER_ORDER = ["tier_a", "tier_b", "tier_c"]


def _bucket_order(b: str) -> int:
    return {
        "0": 0, "<1KB": 1, "1-64KB": 2, "64KB-1MB": 3,
        "1-16MB": 4, "16-256MB": 5, "256MB+": 6,
    }.get(b, 99)


def _parse_run(run_dir: Path) -> dict:
    """Return {tier: {recipe: {...}, hist: {(op,bucket,nranks): count}}}"""
    out: dict[str, dict] = {}
    for tier in TIER_ORDER:
        td = run_dir / f"{tier}_trace"
        csv_path = td / "collective_summary.csv"
        recipe_path = run_dir / "recipe.json"
        if not csv_path.exists():
            continue
        recipe = {}
        if recipe_path.exists():
            try:
                recipe = json.loads(recipe_path.read_text())
            except Exception:
                pass
        hist: Counter = Counter()
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row["opname"], row["size_bucket"], row["nranks"])
                hist[key] += 1
        out[tier] = {"recipe": recipe, "hist": hist, "trace_dir": td}
    return out


def _config_id(run_dir: Path) -> str:
    name = run_dir.name
    # 8gpu_<cfg>_seed42, 8gpu_<cfg>_tier_b, 8gpu_<cfg>_tier_a
    m = re.match(r"8gpu_(.+)_(?:seed\d+|tier_[abc])$", name)
    return m.group(1) if m else name.replace("8gpu_", "")


def main() -> int:
    # Group all 8gpu_* dirs by config_id, merging tier subdirs
    runs_by_cfg: dict[str, dict[str, dict]] = defaultdict(dict)
    for d in sorted(RUNS_ROOT.glob("8gpu_*")):
        if not d.is_dir():
            continue
        cfg = _config_id(d)
        parsed = _parse_run(d)
        for tier, info in parsed.items():
            runs_by_cfg[cfg][tier] = info

    if not runs_by_cfg:
        print(f"WARN: no runs found under {RUNS_ROOT}")
        return 1

    lines: list[str] = []
    lines.append("# Phase 7 NCCL Collective Pattern Catalog")
    lines.append("")
    lines.append(
        "Auto-generated from `phase7_nccl_traffic_catalog/extract_collectives.py` outputs across "
        "all `phase5_vlm_multimodal_sft/runs/8gpu_*/tier_{a,b,c}_trace/collective_summary.csv` "
        "files. **PCIe wallclock is uninterpretable on this hardware; "
        "pattern data (op, size, participants, count) is independent of "
        "physical interconnect and is the deliverable.**"
    )
    lines.append("")

    # Replay priority table
    lines.append("## Replay priority (most realistic first)")
    lines.append("")
    lines.append(
        "| Priority | Config | Tier | GBS | Steps | Trace dir | Total collectives |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|"
    )
    rows = []
    for cfg, tiers in sorted(runs_by_cfg.items()):
        for tier in TIER_ORDER:
            if tier not in tiers:
                continue
            info = tiers[tier]
            recipe = info["recipe"]
            total = sum(info["hist"].values())
            rows.append(
                (
                    TIER_ORDER.index(tier),
                    cfg,
                    tier,
                    recipe.get("global_bs", "?"),
                    recipe.get("steps", "?"),
                    str(info["trace_dir"].relative_to(WORKSPACE)),
                    total,
                )
            )
    rows.sort()
    for i, (_, cfg, tier, gbs, steps, td, total) in enumerate(rows, 1):
        lines.append(
            f"| {i} | `{cfg}` | {tier} | {gbs} | {steps} | `{td}` | {total} |"
        )
    lines.append("")

    # Per-config × tier collective histogram
    lines.append("## Collective histograms per (config, tier)")
    lines.append("")
    for cfg, tiers in sorted(runs_by_cfg.items()):
        lines.append(f"### `{cfg}`")
        lines.append("")
        all_keys = set()
        for tier in TIER_ORDER:
            if tier in tiers:
                all_keys.update(tiers[tier]["hist"].keys())
        if not all_keys:
            lines.append("_No NCCL traces collected._")
            lines.append("")
            continue
        lines.append("| Collective | Size bucket | nranks | Tier A count | Tier B count | Tier C count |")
        lines.append("|---|---|---|---:|---:|---:|")
        sorted_keys = sorted(
            all_keys,
            key=lambda k: (k[0], _bucket_order(k[1]), int(k[2])),
        )
        for k in sorted_keys:
            op, bkt, nr = k
            ta = tiers.get("tier_a", {}).get("hist", {}).get(k, 0)
            tb = tiers.get("tier_b", {}).get("hist", {}).get(k, 0)
            tc = tiers.get("tier_c", {}).get("hist", {}).get(k, 0)
            lines.append(f"| {op} | {bkt} | {nr} | {ta} | {tb} | {tc} |")
        lines.append("")

    # Cross-config summary at Tier A (the headline)
    lines.append("## Cross-config comparison at Tier A (production-standardized)")
    lines.append("")
    lines.append(
        "Which collectives fire under each 3D config, at production "
        "tensor sizes. A blank cell means that collective never appeared "
        "in that config's Tier A trace."
    )
    lines.append("")
    cfg_list = [c for c in sorted(runs_by_cfg) if "tier_a" in runs_by_cfg[c]]
    if cfg_list:
        all_a_keys = set()
        for c in cfg_list:
            all_a_keys.update(runs_by_cfg[c]["tier_a"]["hist"].keys())
        sorted_a = sorted(
            all_a_keys, key=lambda k: (k[0], _bucket_order(k[1]), int(k[2]))
        )
        header = "| op | size | nranks | " + " | ".join(f"`{c}`" for c in cfg_list) + " |"
        sep = "|---|---|---|" + "---|" * len(cfg_list)
        lines.append(header)
        lines.append(sep)
        for k in sorted_a:
            op, bkt, nr = k
            row_cells = []
            for c in cfg_list:
                cnt = runs_by_cfg[c]["tier_a"]["hist"].get(k, 0)
                row_cells.append(str(cnt) if cnt else "")
            lines.append(f"| {op} | {bkt} | {nr} | " + " | ".join(row_cells) + " |")
        lines.append("")
    else:
        lines.append("_No Tier A traces collected yet._")
        lines.append("")

    # Caveats
    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "* **PCIe wallclock is unrepresentative.** Counts and tensor "
        "sizes are framework-determined and portable; latency / "
        "throughput would shift ~10× on NVLink-class interconnects."
    )
    lines.append(
        "* **Tier C is alignment-load (GBS=12).** Counts here are "
        "low-batch and don't reflect production overlap behavior. "
        "Use Tier A for replay decisions."
    )
    lines.append(
        "* **CP=2 traces missing.** kimi_linear's KDA layers' fla-core "
        "kernel doesn't support ring-recurrence over seq-sharded inputs. "
        "See `parallelize.py` CP branch."
    )
    lines.append(
        "* **TP plan is conservative.** Dense MLP only; KDA/MLA/AttnRes "
        "stay replicated. See `apply_tp_kimi_linear` docstring."
    )
    lines.append("")

    CATALOG.write_text("\n".join(lines))
    print(f"Wrote {CATALOG} ({sum(1 for _ in lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
