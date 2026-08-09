#!/usr/bin/env bash
# GPU-side correctness gate for a REFACTORED kimi_k3 tree.
#
# Division of labour: the code-quality cleanup for the upstream PR happens elsewhere; this
# is what checks that the cleanup did not change behaviour. Every gate here failed at least
# once during development, so each one is a real trap rather than a formality, and the
# reason each exists is stated so a future refactor can tell what it would be breaking.
#
#   L1 (~3 min, CPU)   unit suite. Catches structural breakage immediately.
#   L2 (~10 min, GPU)  the five single-axis parallelism cells plus the DEP forward gate.
#                      Catches "a parallelism no longer produces the same arithmetic".
#   L3 (~50 min, GPU)  the full 18-cell matrix plus the DEP/dynamic-CP combined arm.
#                      Catches interaction regressions, including the DEP x EP defect.
#
# Usage: verify_refactor.sh [L1|L2|L3]      (default L2; L3 implies L1 and L2)
#
# Reference values are the ones measured on 2026-08-09 at commit 3d67e3bb5, recorded in
# MATRIX_18_CORRECTED_2026-08-09.md. A refactor is expected to reproduce them EXACTLY at
# step 1 -- these are all --debug.deterministic runs with a fixed seed, so anything else
# means the arithmetic moved.
set -uo pipefail

LEVEL=${1:-L2}
TITAN=${TITAN:-/workspace/torchtitan_attention_residual/torchtitan}
OUT=${OUT:-/workspace/verify_refactor}
FLAVOR=${FLAVOR:-kimi_k3_debugmodel_report_arch}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate
RESULT="$OUT/results.txt"
: > "$RESULT"

port() { echo $((29500 + RANDOM % 900)); }

say() { printf "%s\n" "$*" | tee -a "$RESULT"; }

# --- L1 -------------------------------------------------------------------------------
say "=== L1: unit suite (expect 312 passed, 1 skipped) ==="
timeout 2400 python -m pytest torchtitan/models/kimi_k3/tests/ -q --no-header \
  > "$OUT/l1.log" 2>&1
say "L1 $(grep -oE '[0-9]+ passed.*' "$OUT/l1.log" | tail -1)"
[ "$LEVEL" = L1 ] && exit 0

# --- L2 -------------------------------------------------------------------------------
# Reference step-1 losses, from MATRIX_18_CORRECTED_2026-08-09.md. A single axis at a time,
# so a mismatch localises to that parallelism rather than to a combination.
say ""
say "=== L2: single-axis cells, step-1 loss must match the reference exactly ==="
say "$(printf '%-24s %-12s %s' cell measured reference)"
run_cell() {  # name  reference_step1  args...
  local name="$1" ref="$2"; shift 2
  rm -rf "$OUT/$name"
  timeout 3600 torchrun --nproc_per_node="${NPROC:-8}" --master_port="$(port)" \
    -m torchtitan.train --module kimi_k3 --config "$FLAVOR" --debug.seed 42 \
    --debug.deterministic --metrics.log_freq 1 --training.steps "${STEPS:-2}" \
    --training.global-batch-size 8 "$@" --checkpoint.interval 100000 \
    --dump-folder "$OUT/$name" > "$OUT/$name.log" 2>&1
  local got
  got=$(sed -E 's/\x1b\[[0-9;]*m//g' "$OUT/$name.log" | grep -E 'step: +1 +loss' \
    | grep -vE 'loss: +-' | grep -oE 'loss: *[0-9.]+' | head -1 | tr -dc '0-9.')
  local verdict="MISMATCH"
  [ "$got" = "$ref" ] && verdict=ok
  [ -z "$got" ] && verdict="NO OUTPUT: $(grep -aoE '(RuntimeError|ValueError|AssertionError|NotImplementedError): .{0,50}' "$OUT/$name.log" | head -1)"
  say "$(printf '%-24s %-12s %-12s %s' "$name" "${got:-none}" "$ref" "$verdict")"
  rm -rf "$OUT/$name"
}

NPROC=1 run_cell dp1   12.05342 --parallelism.data_parallel_shard_degree 1
NPROC=2 run_cell fsdp2 12.05033 --parallelism.data_parallel_shard_degree 2
NPROC=2 run_cell cp2   12.03828 --parallelism.data_parallel_shard_degree 1 \
  --parallelism.context_parallel_degree 2
NPROC=2 run_cell tp2   12.06346 --parallelism.data_parallel_shard_degree 1 \
  --parallelism.tensor_parallel_degree 2
NPROC=2 run_cell pp2   12.04891 --parallelism.data_parallel_shard_degree 1 \
  --parallelism.pipeline_parallel_degree 2 --training.local-batch-size 2

# DEP's forward must stay neutral. This is the gate that the tower-split work is held to,
# and it needs a SHARED SEED CHECKPOINT: DEP changes the stage split, so cold-start arms
# consume RNG differently and are not comparable. Getting this wrong produced a misleading
# 0.05 once, and a misleading step-1 offset a second time.
say ""
say "=== L2b: DEP forward neutrality at pp4, n_vit 1 vs 2, from a shared seed ==="
SEED="$OUT/seed"; rm -rf "$SEED"
timeout 900 torchrun --nproc_per_node=1 --master_port="$(port)" -m torchtitan.train \
  --module kimi_k3 --config "$FLAVOR" --debug.seed 42 --debug.deterministic \
  --training.steps 1 --training.global-batch-size 8 --training.local-batch-size 8 \
  --parallelism.data_parallel_shard_degree 1 --checkpoint.create_seed_checkpoint \
  --checkpoint.enable --dump-folder "$SEED" > "$OUT/seed.log" 2>&1
for N in 1 2; do
  R="$OUT/dep$N"; rm -rf "$R"; mkdir -p "$R"; cp -r "$SEED/checkpoint" "$R/checkpoint"
  KIMI_VIT_DEP=1 KIMI_VIT_DEP_STAGES=$N timeout 1800 torchrun --nproc_per_node=4 \
    --master_port="$(port)" -m torchtitan.train --module kimi_k3 --config "$FLAVOR" \
    --debug.seed 42 --debug.deterministic --training.steps 2 \
    --training.global-batch-size 8 --training.local-batch-size 8 \
    --parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 4 \
    --checkpoint.enable --checkpoint.interval 100000 --dump-folder "$R" \
    > "$OUT/dep$N.log" 2>&1
  say "$(printf 'n_vit=%-3s wiring=%-3s step1=%s (all n_vit must agree; 12.07418 on 2026-08-09)' \
    "$N" "$(grep -ac 'vision stage wiring' "$OUT/dep$N.log")" \
    "$(sed -E 's/\x1b\[[0-9;]*m//g' "$OUT/dep$N.log" | grep -E 'step: +1 +loss' | grep -vE 'loss: +-' | grep -oE 'loss: *[0-9.]+' | head -1 | tr -dc '0-9.')")"
  rm -rf "$R"
done
rm -rf "$SEED"
[ "$LEVEL" = L2 ] && exit 0

# --- L3 -------------------------------------------------------------------------------
say ""
say "=== L3: full 18-cell matrix ==="
say "The three TP+CP cells need KIMI_VIT_DYNAMIC_CP=0 until the gather-KV/DTensor defect"
say "is fixed; run13_flav.sh does NOT set it, so those three are expected to fail here."
FLAVOR=$FLAVOR OUT="$OUT/mx_a" STEPS=10 bash \
  "$(dirname "$0")/run13_flav.sh" >> "$OUT/l3a.log" 2>&1
FLAVOR=$FLAVOR OUT="$OUT/mx_b" STEPS=10 bash \
  "$(dirname "$0")/run_maxdeg.sh" >> "$OUT/l3b.log" 2>&1
bash "$(dirname "$0")/collect13.sh" "$OUT/mx_a" 10 | tee -a "$RESULT"
say ""
say "=== L3b: DEP x EP is KNOWN BROKEN -- NaN at step 2, see MATRIX_18_CORRECTED ==="
say "If this cell now SURVIVES, that is news and the limitation note should be revisited."
R="$OUT/depep"; rm -rf "$R"
KIMI_VIT_DEP=1 KIMI_VIT_SIDE_STREAM=1 timeout 1800 torchrun --nproc_per_node=8 \
  --master_port="$(port)" -m torchtitan.train --module kimi_k3 --config "$FLAVOR" \
  --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 4 \
  --training.global-batch-size 8 --training.local-batch-size 2 \
  --parallelism.data_parallel_shard_degree 2 --parallelism.expert_parallel_degree 2 \
  --parallelism.pipeline_parallel_degree 2 --parallelism.context_parallel_degree 2 \
  --checkpoint.interval 100000 --dump-folder "$R" > "$OUT/depep.log" 2>&1
say "dep+ep exit=$? $(grep -aoE 'Loss is not finite.{0,40}' "$OUT/depep.log" | head -1)"
rm -rf "$R"
say ""
say "=== DONE ==="
