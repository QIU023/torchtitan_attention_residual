# Additional issues found during phase 6/7 8-GPU work

This folder collects upstream / cross-cutting issues that surfaced
while validating multi-D parallelism on the 8-GPU PCIe box. Each
issue gets its own markdown with: symptom, reproduction, root-cause
hypothesis, severity, candidate fixes, and status (filed upstream
yes/no).

These are **not blockers for the AttnRes / cache adapter PR**. They
are observations about the broader torchtitan / fla-core / pytorch
ecosystem that are worth filing as separate RFCs / issues.

## Index

| # | File | Surface | Severity | Status |
|---|---|---|---|---|
| 1 | [`torchtitan_pp_microbatch_backward_graph.md`](torchtitan_pp_microbatch_backward_graph.md) | "Trying to backward through the graph a second time" under PP+Interleaved1F1B with LOCAL_BS≥3 and V≥2 | High (blocks production-batch PP+TP) | Documented; sub-agent investigation in flight |
| 2 | [`kimi_kda_head_dim_73_blackwell.md`](kimi_kda_head_dim_73_blackwell.md) | Kimi paper Table 2 produces `head_dim ≈ 73-75` (prime / non-multiple of 64/128); on Blackwell sm_120 this defeats tensor-core tile alignment, MFU stays 0.21% | Medium (perf only; correctness fine) | Documented; out-of-scope for AttnRes PR (paper-faithful arch choice) |

## Filing protocol

For each issue we want to upstream:
1. Reproduce on a stripped-down example (no AttnRes, no kimi_linear-specific)
2. Open issue / RFC on the appropriate repo (pytorch/torchtitan, fla-org/flash-linear-attention, pytorch/pytorch)
3. Link the doc here back to the upstream issue URL
