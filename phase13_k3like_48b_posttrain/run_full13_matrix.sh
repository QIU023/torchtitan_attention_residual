#!/usr/bin/env bash
# The 13-leg D matrix from BASELINE_MATRIX_2026-08-03.md, re-run against the
# current tree. Same fixture: diag_4l_moe_depth, 3 steps, seed 42,
# deterministic, global batch 8, cold init per layout (no seed checkpoint --
# that is what the recorded baseline used).
#
# Collector rules this project has had to re-learn:
#   - strip ANSI before matching (a torch upgrade once made all 13 read FAILED)
#   - drop PP's negative placeholders BEFORE dedup, and dedup on the whole
#     line, not the step number (dedup-then-filter once deleted every row)
set -uo pipefail

TITAN=${TITAN:-/workspace/torchtitan_attention_residual/torchtitan}
OUT=${OUT:-/tmp/full13}
STEPS=${STEPS:-3}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

FLAVOR=kimi_k3_mini_diag_4l_moe_depth
BASE="--module kimi_k3 --config $FLAVOR --training.seq_len 512 --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1 --training.steps $STEPS \
 --training.global-batch-size 8"

losses() { sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "step: +[0-9]+ +loss" \
  | sed -E 's/.*(step: +[0-9]+ +loss: +[-0-9.]+).*/\1/' \
  | grep -vE 'loss: +-' | sort -u; }
fails() { sed -E 's/\x1b\[[0-9;]*m//g' | grep -oiE \
  "(RuntimeError|ValueError|AssertionError|KeyError|NotImplementedError|OutOfMemoryError|TypeError): .{0,70}" \
  | head -1; }

PORT=48300
run() {
  local name="$1" ngpu="$2"; shift 2
  PORT=$((PORT+1))
  local out uniq n
  # Wipe the per-leg dump folder: a leftover checkpoint from an earlier run
  # makes the trainer RESUME, so a 10-step run after a 3-step one produces 7
  # rows and the collector calls it a failure. Third time this project has
  # been bitten by checkpoint carry-over.
  rm -rf "$OUT/$name"
  out=$(CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((ngpu-1))) timeout 2400 torchrun \
        --nproc_per_node="$ngpu" --master_port=$PORT -m torchtitan.train \
        $BASE "$@" --dump-folder "$OUT/$name" 2>&1)
  uniq=$(echo "$out" | losses)
  n=$(echo "$uniq" | grep -c "step:")
  if [ "$n" -eq "$STEPS" ]; then
    printf "%-22s %s\n" "$name" \
      "$(echo "$uniq" | grep -oE 'loss: +[0-9.]+' | grep -oE '[0-9.]+' | tr '\n' ' ')"
  else
    printf "%-22s FAIL (%d/%d rows) %s\n" "$name" "$n" "$STEPS" "$(echo "$out" | fails)"
  fi
}

PPB="--training.local-batch-size 2"   # PP needs microbatches >= stages
D=--parallelism.data_parallel_shard_degree
T=--parallelism.tensor_parallel_degree
P=--parallelism.pipeline_parallel_degree
C=--parallelism.context_parallel_degree
E=--parallelism.expert_parallel_degree

echo "### singles ###"
run dp1                 1 $D 1
run fsdp2               2 $D 2
run pp2                 2 $PPB $D 1 $P 2
run cp2                 2 $D 1 $C 2
run tp2                 2 $D 1 $T 2

echo "### 3 of 4 ###"
run fsdp2_tp2_pp2       8 $PPB $D 2 $T 2 $P 2
run fsdp2_tp2_cp2       8 $D 2 $T 2 $C 2
run tp2_pp2_cp2         8 $PPB $D 1 $T 2 $P 2 $C 2
run fsdp2_pp2_cp2       8 $PPB $D 2 $P 2 $C 2

echo "### EP on ###"
run ep2_fsdp2           2 $D 2 $E 2
run ep2_fsdp2_tp2_pp2   8 $PPB $D 2 $E 2 $T 2 $P 2
run ep2_fsdp2_tp2_cp2   8 $D 2 $E 2 $T 2 $C 2
run ep2_fsdp2_pp2_cp2   8 $PPB $D 2 $E 2 $P 2 $C 2
echo "### DONE ###"
