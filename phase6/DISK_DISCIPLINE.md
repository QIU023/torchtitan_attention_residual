# Disk Discipline (phase 6 +)

This file documents disk-management rules for long-running training in
`phase5/runs/`. Failure to follow these rules has filled the 200 GB
volume twice during v10/v11/v12 retry-loop runs, halting training and
wasting GPU rental.

## Failure mode

Auto-retry loops re-launch torchrun on each crash. Each attempt:
- Re-allocates 16 GB of FSDP/DCP checkpoint shards (one per save).
- Re-allocates NCCL trace logs in `tier_b_trace/` (~18 GB raw across 8
  ranks for a 50-step trace window).
- Allocates additional TB events in `tb/`, ~1 MB but accumulates.

Without explicit cleanup between attempts, 25 attempts → 25 × 18 GB
= 450 GB of trace alone. With KEEP_K=2 ckpt + 25 trace dirs, the 200
GB volume hits ENOSPC fast — at which point even Bash tool calls fail
because their stdout cannot be written to `/tmp`.

Both v10 (June 2026 incident, ENOSPC after 8 alignment runs) and v12
(this incident) hit this exact pattern.

## Mandatory rules for every retry-loop launcher

1. **Trace only on first attempt.** Set `TRACE_TIER=tier_b` only when
   `attempt == 1`; unset on retries. We don't need 25 copies of the
   same NCCL pattern.

2. **Pre-flight disk check.** Before each attempt, verify free space
   ≥ `(KEEP_K + 1) * ckpt_size_estimate`. For a 1.2B model this is
   `(2+1) * 16 = 48 GB` minimum. Abort the loop with a loud error
   if not met.

3. **Clean tier_b_trace between attempts.** After each crashed attempt,
   delete `OUT_DIR/tier_b_trace/nccl-rank-*.log` if `attempt > 1`. The
   first attempt's trace is preserved as the canonical fabric profile
   sample.

4. **KEEP_K capped at 2.** Anything higher multiplies disk pressure
   linearly. Use `SAVE_FREQ=200` (frequent checkpoints) with `KEEP_K=2`
   so worst-case rollback is ≤200 steps without growing disk usage.

5. **Don't pre-emptively cache compressed CSV alongside raw**. After
   `phase7/extract_collectives.py`, immediately delete the raw
   `nccl-rank-*.log` files; the `collective_summary.csv.gz` (~210 MB
   for a 13-attempt run) is the long-lived artifact.

## Pre-flight check helper (template)

```bash
# Inside the retry loop, before each attempt
free_gb=$(df -BG --output=avail "$OUT_DIR" | tail -1 | tr -d 'G ')
required_gb=$((48))  # 3 * 16 GB ckpts
if [[ "$free_gb" -lt "$required_gb" ]]; then
    echo "[$(date)] DISK ABORT: $free_gb GB free < $required_gb GB required"
    echo "[$(date)] Free space first; loop will not proceed"
    break
fi
```

## Why monitor on the AGENT side too

The Claude agent observability layer must check `df -h /root` at every
inflection point during long autonomous windows: after each ckpt save
boundary, after each retry, when starting a new heavy task. This is
encoded in the `feedback_disk_monitoring` memory file. The launcher
code path is the second line of defense; agent-level monitoring is
the first.
