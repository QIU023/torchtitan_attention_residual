# Checkpoint + disk discipline rules v2 (HARD ENFORCEMENT)

**v1** (`CHECKPOINT_RULES.md`) was advisory and was repeatedly violated.
**This is v2 — hard rules with enforcement scripts.** Every launcher MUST
source the helpers below; if a launcher doesn't, the orchestrator MUST
refuse to run it.

## Why v2 exists — 2026-05-11 incident

Overnight chain (SFT step 5500→7000 + GRPO 1500 + PP sweep) filled the
container disk to 100% during the L16 PP=4 V=4 adapter rerun. Direct
causes:

1. SFT kept **5** ckpts (`step-5000/5500/6000/6500/7000`) at ~15 GB each =
   75 GB just from SFT, despite v1 saying `keep_latest_k=2`.
2. L16 sweep runs (6 × 2-min smoke runs) saved per-run ckpt dirs.
3. No mid-chain disk monitor; the orchestrator only noticed when the
   last `Bash` call returned ENOSPC.
4. Partial recovery cost: GRPO Stage C trial output + L16 PP=4 V=4
   adapter run output were both truncated; SFT crash-resume logic
   masked the early-warning signs by re-saving from intermediate ckpts.

## The hard rules

### Rule 1 — `keep_latest_k=1` default; `k>1` requires explicit annotation

| Run type | `KEEP_K` | `SAVE_FREQ` | `CHECKPOINT_ENABLED` |
|---|---|---|---|
| Alignment / NCCL trace / smoke / sweep / pressure-test | **N/A** | **N/A** | **0** (never save) |
| Long pretrain (≥ 5000 steps) | **1** | 1000 | 1 |
| SFT (1-2 epochs) | **1** | 500 | 1 |
| Mission-critical (need recoverable mid-step) | **2** | 500 | 1, **with comment in launcher explaining why** |

**Default in all launchers MUST be `KEEP_K=1`.** Any launcher setting
`KEEP_K=2` or higher MUST include an inline comment justifying the
extra disk cost.

### Rule 2 — OUT_DIR name pattern auto-disables checkpointing

Any launcher whose `OUT_DIR` matches one of these patterns MUST force
`CHECKPOINT_ENABLED=0`, regardless of what the user set:

- `*sweep*`
- `*pressure*`
- `*smoke*`
- `*align*` (alignment)
- `*trace*` (NCCL trace tiers)
- `*tier_[abcd]*` (named trace tiers)
- `*_naive_*` / `*_adapter_*` (paired numerical-alignment runs)

Source `phase6/preflight_disk.sh` at launcher start; it implements this
pattern check.

### Rule 3 — Mandatory pre-launch `df` check

`phase6/preflight_disk.sh` MUST run before any torchrun and refuse to
launch if free disk < 80 GB. No exceptions. The script also handles
auto-prune of stale ckpts from prior runs.

### Rule 4 — Mandatory disk watchdog during long runs

Any run estimated > 1 hour MUST start `phase6/disk_watchdog.sh` as a
background process. The watchdog polls `df -h` every 60s and
**`kill -9`s the entire torchrun process group** if disk used > 90%.

Watchdog PID is written to `${OUT_DIR}/.watchdog.pid` so the launcher
can clean up on exit via `trap`.

### Rule 5 — Auto-prune before any new run

`phase6/preflight_disk.sh` ALSO does:

```bash
# Delete stale checkpoint dirs from prior runs that we no longer need.
# Heuristic: any run dir whose name matches the disabled-ckpt patterns
# (Rule 2) is fair game.
for dir in $(find "$RUNS_ROOT" -maxdepth 2 -type d -name "checkpoint" \
             -path "*sweep*" -o -path "*pressure*" -o -path "*smoke*" \
             -o -path "*align*" -o -path "*trace*" -o -path "*_naive_*" \
             -o -path "*_adapter_*"); do
    echo "[preflight] auto-prune $dir"
    rm -rf "$dir"
done
```

This is run unconditionally before every launch. **Sweep / smoke /
alignment ckpts are throwaway; deleting them on next launch is correct
behavior.**

### Rule 6 — Chain orchestrators emit disk reports

Any multi-stage orchestrator (e.g. `phase11/run_phase6_closure_then_pretrain.sh`)
MUST `echo "[disk] $(df -h /)"` at every stage boundary into the chain
log so post-mortem can identify exactly which stage exhausted the disk.

### Rule 7 — DCP saves default to model-only

For runs whose downstream consumer is only inference (DCP→HF →
SGLang), use `--checkpoint.exclude_states` to drop optim state
(~3× the model weight bytes). Only keep optim state when the run's
own ckpt is the resume point for continued training.

### Rule 8 — Sweep orchestrators emit a `manifest.json`

Multi-run scripts (e.g. PP shape sweep, ablation grids) MUST write
`${SWEEP_OUT_ROOT}/manifest.json` listing every run's `OUT_DIR` and
a `preserve` flag (`true` / `false`). At sweep completion the
orchestrator runs `rm -rf` on every `preserve=false` dir.

## Enforcement helpers (in this folder)

| Script | Purpose |
|---|---|
| `preflight_disk.sh` | Pre-launch disk check + auto-prune of stale sweep/smoke/trace ckpts. **Sources at every launcher's top.** Exits 1 if free < 80 GB. |
| `disk_watchdog.sh` | Background poller — `kill -9` torchrun PGID if disk used > 90%. PID written to `${OUT_DIR}/.watchdog.pid`. |

## Auditing existing launchers

Run `grep -l "torchrun" phase*/run*.sh phase*/launch*.sh` and check
each one:

- [ ] sources `phase6/preflight_disk.sh`
- [ ] starts `phase6/disk_watchdog.sh` if estimated runtime > 1h
- [ ] uses `KEEP_K=1` by default (or has a comment justifying `k>1`)
- [ ] checks `OUT_DIR` name pattern → forces `CHECKPOINT_ENABLED=0` for
      sweep / smoke / trace / alignment

Any launcher failing the audit MUST be patched before its next run.

## Action items (one-time, after this doc lands)

1. Patch every existing `phase[3-11]/run*.sh` and `phase[3-11]/launch*.sh`
   to source `preflight_disk.sh`. (Tedious — write a `sed` one-liner.)
2. Default-`KEEP_K=1` audit: grep for `--checkpoint.keep_latest_k` and
   `keep_latest_k=` across all launchers; change to 1 unless justified.
3. Re-run `phase7/auto_publish_watcher.sh` to clean any stale sweep ckpts
   created before this rules doc landed.

## Why disk discipline is part of the project's deliverables

For ML infra / training-infra interview audiences, this rules doc is
itself **a signal**: it shows you've absorbed a real production incident
(disk-full mid-overnight chain costing ~7 GPU-h) and converted it into
mechanical enforcement, not a memo. That's the muscle ML infra teams
actually care about.
