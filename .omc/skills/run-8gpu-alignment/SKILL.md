---
name: run-8gpu-alignment
description: Run the phase-6 8-GPU alignment matrix (FSDP × PP × TP × EP variants) and summarize the per-cell PASS/FAIL.
triggers:
  - "alignment matrix"
  - "phase 6 alignment"
  - "8gpu alignment"
agent: executor
model: sonnet
---

# run-8gpu-alignment

## Purpose
Drive `phase6/run_8gpu_alignment_matrix.sh` end-to-end on the local 8-GPU box
and produce a single PASS/FAIL summary across the variants currently considered
runnable (A2 is the headline; A3/A6/all4d/noPP are documented upstream-blocked
in `phase6/README.md` — do not attempt them unless the user says the upstream
fix landed).

## Workflow

1. **Pre-flight**
   - Confirm the box has 8 GPUs: `nvidia-smi --query-gpu=index --format=csv` (already allowlisted in `.claude/settings.local.json`).
   - Confirm `torchtitan/` submodule is on the expected commit:
     `git -C torchtitan rev-parse HEAD` against the SHA recorded in the most
     recent `phase 6:` commit message.
   - Confirm `attnres` conda env is active.

2. **Run the matrix**
   - `bash phase6/run_8gpu_alignment_matrix.sh 2>&1 | tee phase6/orchestrator_8gpu.log`
   - Run sequentially (the script already serializes by GPU contention); do
     **not** parallelize across variants on a single 8-GPU box.

3. **Parse per-variant reports**
   - For each variant `V` in `{a2_fsdp2_pp4, a3_fsdp2_pp2_tp2, all4d_fsdp2_pp2_tp2_ep2, noppc_fsdp4_tp2_ep2}`:
     - Read `phase6/alignment_8gpu_${V}.txt`.
     - Extract the final-step loss delta vs the recorded baseline.
     - Mark PASS if `|Δ| ≤ 0.13` (the nondeterminism band from phase 3),
       FAIL otherwise.

4. **Write the summary**
   - Append a short table to `phase6/SESSION_8GPU_summary.md` under a new
     dated heading.
   - Skip variants the user has flagged upstream-blocked unless the user
     re-enabled them explicitly this session.

5. **Commit (only if user asks)**
   - Stage the new `alignment_8gpu_*.txt` and the summary append.
   - Commit message style: `phase 6: <variant> alignment <PASS|FAIL>, Δ=<value>`
     (match the `24562fd` / `7ff7cd7` commit style).

## Usage

```text
/oh-my-claudecode:run-8gpu-alignment
```

Optional arg: a comma-separated variant filter, e.g. `a2_fsdp2_pp4,noppc_fsdp4_tp2_ep2`.

## Configuration

- `MAX_RETRIES` (default 1): re-launch a single variant once if NCCL hangs.
- `BASELINE_SHA`: torchtitan submodule SHA the deltas are computed against;
  defaults to whatever is checked out.

## Do not

- Do not edit `phase6/run_8gpu_alignment_matrix.sh` from this skill — that
  script is the source of truth; if it's wrong, fix it directly and commit.
- Do not delete prior `alignment_8gpu_*.txt` outputs; they're the only
  provenance for past matrix runs.
