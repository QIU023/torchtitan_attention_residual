#!/usr/bin/env bash
# 13-leg matrix on kimi_k3_debugmodel_pr_4025, the #4025 twin.
#
# Deliberately does NOT override seq_len or local_batch_size: the flavor now
# carries #4025's own values (256 / 1), and local_batch_size 1 against global
# batch 8 means gradient accumulation is on by default -- which is the
# configuration that exposed the zero-sentinel CP defect, so the matrix should
# run it rather than avoid it.
set -uo pipefail

TITAN=${TITAN:-/workspace/torchtitan_attention_residual/torchtitan}
OUT=${OUT:-/tmp/twin13}
STEPS=${STEPS:-3}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

FLAVOR=kimi_k3_debugmodel_pr_4025
BASE="--module kimi_k3 --config $FLAVOR --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1 --training.steps $STEPS \
 --training.global-batch-size 8 $EXTRA"

losses() { sed -E 's/\x1b\[[0-9;]*m//g' | grep -E "step: +[0-9]+ +loss" \
  | sed -E 's/.*(step: +[0-9]+ +loss: +[-0-9.]+).*/\1/' \
  | grep -vE 'loss: +-' | sort -u; }
fails() { sed -E 's/\x1b\[[0-9;]*m//g' | grep -oiE \
  "(RuntimeError|ValueError|AssertionError|KeyError|NotImplementedError|OutOfMemoryError|TypeError|InternalError): .{0,70}" \
  | head -1; }

PORT=48700
run() {
  local name="$1" ngpu="$2"; shift 2
  PORT=$((PORT+1))
  local out uniq n
  rm -rf "$OUT/$name"   # leftover checkpoints make a run RESUME, not restart
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
