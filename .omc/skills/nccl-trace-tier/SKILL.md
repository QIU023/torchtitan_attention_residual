---
name: nccl-trace-tier
description: Run a phase-7 NCCL trace tier (a/b/c), extract the collective summary, and refresh pattern_catalog.md.
triggers:
  - "nccl trace"
  - "trace tier"
  - "pattern catalog"
agent: executor
model: sonnet
---

# nccl-trace-tier

## Purpose
Wrap the phase-7 trace pipeline so a single invocation produces the raw NCCL
logs, the parsed `collective_summary.csv`, and an updated
`phase7_nccl_traffic_catalog/pattern_catalog.md` for the requested tier.

Tier semantics (from `phase7_nccl_traffic_catalog/README.md`):
- `tier_a` — quick smoke, ~10 MB raw logs per dir, minutes.
- `tier_b` — standard catalog input, ~50 MB per dir.
- `tier_c` — full ablation, ~200 MB per dir, hour+.

Raw `nccl-rank-*.log` files and `collective_summary.csv` are gitignored
(`.gitignore` lines under `phase5_vlm_multimodal_sft/runs/**/tier_*_trace/`); only
`pattern_catalog.md` and the `train.log` tail are committed.

## Workflow

1. **Pick the tier and the run dir**
   - Default tier: `tier_a` (cheapest).
   - Default run dir: most recent under `phase5_vlm_multimodal_sft/runs/8gpu_*/` that has no
     existing `tier_${TIER}_trace/`. Ask the user if ambiguous.

2. **Launch the trace**
   - `TRACE_TIER=tier_${TIER} bash phase6_upstream_pr_prep/launch_8gpu_mm.sh <run-dir>`
     OR `bash phase7_nccl_traffic_catalog/run_tier_b_a_traces.sh` for the canned b→a sequence.
   - Tee output to `phase7_nccl_traffic_catalog/orchestrator_tiers.log`.

3. **Extract collectives**
   - `python phase7_nccl_traffic_catalog/extract_collectives.py <run-dir>/tier_${TIER}_trace/`
   - This writes `collective_summary.csv` (gitignored, large — that's fine).

4. **Rebuild the catalog**
   - `python phase7_nccl_traffic_catalog/build_pattern_catalog.py`
   - Diff `phase7_nccl_traffic_catalog/pattern_catalog.md` vs the prior version; if the diff is
     non-trivial (new histogram bucket, new collective op type, distribution
     shift > 5%), summarize the delta in the user-visible response.

5. **Commit (only on user confirmation)**
   - Stage `phase7_nccl_traffic_catalog/pattern_catalog.md` and the run dir's `train.log`.
   - Commit style: `phase 7: <tier> trace for <run-dir-tag>, <one-line delta>`.

## Usage

```text
/oh-my-claudecode:nccl-trace-tier                       # tier_a, latest run
/oh-my-claudecode:nccl-trace-tier tier_b                # explicit tier
/oh-my-claudecode:nccl-trace-tier tier_a 8gpu_a2_fsdp2_pp4_tier_a   # explicit run
```

## Do not

- Do not commit `nccl-rank-*.log`, `nsys-*`, or `collective_summary.csv` —
  they're gitignored for size reasons and reproducible from `extract_collectives.py`.
- Do not run `tier_c` without confirming the user has > 1 hour of GPU time
  budgeted; it serializes the box.
- Do not edit `extract_collectives.py` or `build_pattern_catalog.py` from
  this skill; fix them directly and commit if needed.
