# Checkpoint + trace storage rules (Phase 6/7 8-GPU box)

These rules emerged from a real incident on 2026-05-03: alignment + Tier
B/A trace runs all had `--checkpoint.enable` on, which made torchtitan
allocate disk for state_dict serialization. Even with `interval=999999`,
several runs ended up writing a `step-500` (or step-50/100) DCP shard
~15 GB each. Eight such runs accumulated ~120 GB of zero-value ckpt
artifacts, filled the 200 GB disk, and crashed v10 at step 1000 ckpt
save (ENOSPC). 7 hours of GPU time + ~$20 wasted.

## The rules

### 1. Run-type → checkpoint policy

| Run type | Example | Disk policy |
|---|---|---|
| **Alignment smoke** (≤ 1000 steps, output = loss curve + NCCL trace) | B0/A2/A3 alignment, control configs | **NO checkpoint at all** — pass the launcher with `STUDENT_CKPT=...` for init load only; do NOT enable `--checkpoint.enable`. Skip the `--checkpoint.interval` flag entirely. |
| **NCCL trace tier** (Tier A/B production-load profiling) | b0/a2/a3 tier_a / tier_b | **NO checkpoint** — same as alignment. The trace is the deliverable; intermediate model state is throwaway. |
| **Long-running pretrain** (5000+ steps, output = trained model) | v10, future v11 | **`keep_latest_k=2 interval=500`** — exactly two rolling ckpts. Confirm before launch that `(2 × ckpt_size) + safety_margin < free_disk`. |
| **Production multimodal pretrain** (10000+ steps with crash-resilience need) | future v11+ | Same as long-running, plus a separate "best by val loss" promotion step that copies the chosen ckpt to a different dir before the rolling-k overwrites it. |

### 2. Trace capture is the deliverable for short runs

For alignment + trace tier runs, **the value lives in**:
- `tb/events.out.tfevents.*` (loss curve)
- `tier_*_trace/nccl-rank-*.log` (NCCL collective traces)
- `tier_*_trace/collective_summary.csv` (parsed)
- `recipe.json` (mesh + recipe metadata)

That's all. The model `checkpoint/step-N/__M_0.distcp` files are pure
disk drag — they're never used downstream and never re-loaded.

### 3. Compress + commit + push immediately after each run

Replace the prior pattern (run → archive at end → manual push) with:

```
run completes → phase7/pack_traces.sh → phase7/publish_archive.sh
              → automatic git commit + push to GitHub
              → delete the run's checkpoint dir if it ever wrote one
```

`phase7/auto_publish_watcher.sh` already does the publish part; add
post-publish cleanup to it (rm any `checkpoint/` under
`phase5/runs/8gpu_*` whose recipe says it's an alignment or trace run).

### 4. Pre-launch disk check

Add to `phase6/launch_8gpu_mm.sh`: at script start, if
`CHECKPOINT_ENABLED=1` and `df -h /` reports < 30 GB free, refuse to
launch with a clear "free disk first" error. This is a guard against
exactly the incident this doc captures.

### 5. ckpt size budgeting

Kimi-Linear AttnRes 436M ckpt size at our shard layout:
- single dcp shard ~3.8 GB × 4 ranks = ~15 GB / step
- with `keep_latest_k=2` rolling = 30 GB occupied by ckpts at any time
- 200 GB overlay → comfortable budget = ~150 GB of ckpts max → up to 5
  concurrent long-pretrain runs (rolling k=2 each), or 10 historical
  `best-by-val` copies. Anything beyond that needs to be pushed to
  external storage and deleted.

### 6. What the launcher should set

`phase6/launch_8gpu_mm.sh` env knobs to make the policy automatic:

| Run type | Env to set |
|---|---|
| alignment / trace tier | `CHECKPOINT_ENABLED=0` (new flag — skip `--checkpoint.enable`) |
| long pretrain | `CHECKPOINT_ENABLED=1 SAVE_FREQ=500 KEEP_K=2` |

The current launcher unconditionally enables checkpoint via the
`CKPT_ARGS` block. **This must be made conditional.** Patch:

```bash
CKPT_ARGS=""
if [[ "${CHECKPOINT_ENABLED:-0}" == "1" && -d "${STUDENT_CKPT}" ]]; then
    CKPT_ARGS="--checkpoint.enable \
               --checkpoint.initial_load_path ${STUDENT_CKPT} \
               --checkpoint.initial_load_model_only \
               --checkpoint.interval ${SAVE_FREQ} \
               --checkpoint.keep_latest_k ${KEEP_K}"
elif [[ -d "${STUDENT_CKPT}" ]]; then
    # Init-load-only (no further saves)
    CKPT_ARGS="--checkpoint.enable \
               --checkpoint.initial_load_path ${STUDENT_CKPT} \
               --checkpoint.initial_load_model_only \
               --checkpoint.interval 999999999"
fi
```

(Alignment runs still need `initial_load_path` to load the phase4 init,
but they should never save a new checkpoint themselves. The `enable`
flag has to stay on for the trainer's `state_dict_load` to fire.
`interval=999999999` ensures no save-during-training even if enabled.
This is a torchtitan-side limitation; ideally the trainer would expose
a "load-only never-save" mode but doesn't today.)

Better long-term fix: expose a real `--checkpoint.load_only` flag in
torchtitan core. Out of scope for this experiment.

### 7. Incident postmortem (2026-05-03)

| Field | Value |
|---|---|
| Trigger | v10 attempt-3 step 1000 ckpt save |
| Surface error | `ENOSPC: no space left on device` |
| Root cause | 8 prior alignment/trace runs each left a `step-500` DCP shard ~15 GB on disk; total 120 GB; v10 needed another 30 GB at step 1000 save → 150 GB > 40 GB free |
| Lost work | v10 step 0 → 1000 (~1.7 h GPU); cumulative across earlier failed v10 attempts ~7 h GPU |
| Cost estimate | ~$20 cloud GPU |
| Fix forward | rules above; prune existing `8gpu_*/checkpoint/` dirs before next run; auto-cleanup added to publish_archive.sh |

### 8. Action items immediately after this doc

1. `rm -rf phase5/runs/8gpu_*/checkpoint/` — frees ~120 GB
2. Patch `phase6/launch_8gpu_mm.sh` per §6
3. Update `phase6/run_v10_pretrain.sh` to set `CHECKPOINT_ENABLED=1
   SAVE_FREQ=500 KEEP_K=2` (its current behavior)
4. Update `phase6/run_a3_then_v10_3d.sh` (if it ever re-runs alignment)
   to set `CHECKPOINT_ENABLED=0`
5. Update `phase7/auto_publish_watcher.sh` to nuke any
   `8gpu_*_seed*/checkpoint` dir after publish
6. Re-launch v10 with the corrected launcher
