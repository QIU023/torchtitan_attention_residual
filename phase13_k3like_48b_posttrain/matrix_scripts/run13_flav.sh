#!/usr/bin/env bash
# 13-leg matrix on the #4025 twin, with the small legs run concurrently on
# disjoint GPU sets and the 8-GPU legs serial.
#
# Built-in control: steps 1-3 of a 10-step run must match the 3-step serial
# matrix bit-for-bit. If they do not, concurrency perturbed something and the
# run is not comparable -- check before reading anything else.
set -uo pipefail

TITAN=${TITAN:-/workspace/torchtitan_attention_residual/torchtitan}
OUT=${OUT:-/tmp/twinpar}
STEPS=${STEPS:-10}
EXTRA=${EXTRA:-}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

FLAVOR=${FLAVOR:-kimi_k3_debugmodel_pr_4025}
BASE="--module kimi_k3 --config $FLAVOR --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1 --training.steps $STEPS \
 --training.global-batch-size 8 $EXTRA"

PPB="--training.local-batch-size 2"
D=--parallelism.data_parallel_shard_degree
T=--parallelism.tensor_parallel_degree
P=--parallelism.pipeline_parallel_degree
C=--parallelism.context_parallel_degree
E=--parallelism.expert_parallel_degree

# launch <name> <gpu-list> <port> <args...>   -- writes raw log, no parsing
launch() {
  local name="$1" gpus="$2" port="$3"; shift 3
  local n; n=$(awk -F, '{print NF}' <<<"$gpus")
  # Retry on a fresh port when the assigned one is still held. The ports here are
  # fixed per cell so concurrent cells cannot collide with each other -- but a
  # socket from a PREVIOUS matrix run sits in TIME_WAIT, and back-to-back runs hit
  # that constantly: three consecutive 18-cell runs each lost exactly one cell
  # this way (ep2_fsdp2, tp2, fsdp2_pp2_cp2). Without this the cell reports FAIL
  # and reads exactly like a numerical regression. run_maxdeg.sh already retried;
  # this half of the matrix did not.
  local attempt use_port
  for attempt in 1 2 3; do
    use_port="$port"
    [ "$attempt" -gt 1 ] && use_port=$((port + 400 * (attempt - 1)))
    rm -rf "$OUT/$name"
    CUDA_VISIBLE_DEVICES="$gpus" timeout 7200 torchrun \
      --nproc_per_node="$n" --master_port="$use_port" ${ENTRY:--m torchtitan.train} \
      $BASE "$@" --dump-folder "$OUT/$name" > "$OUT/$name.log" 2>&1
    grep -q EADDRINUSE "$OUT/$name.log" || break
    echo "  ($name: port $use_port in use, retrying)" >&2
  done
  # Drop this cell's checkpoint as soon as the cell is done. A 100-step cell
  # leaves ~700 MB of it and nothing ever reads it; eighteen cells filled the disk
  # and took bash down with it. tb/ stays -- stdout carries five significant
  # digits and a real loss comparison has to come from the TensorBoard record.
  rm -rf "$OUT/$name/checkpoint"
}

echo "### batch A (concurrent, 2 GPUs each) ###"
launch fsdp2 0,1 49101 $D 2 &
launch pp2   2,3 49102 $PPB $D 1 $P 2 &
launch cp2   4,5 49103 $D 1 $C 2 &
launch tp2   6,7 49104 $D 1 $T 2 &
wait

echo "### batch B (concurrent) ###"
launch dp1       0   49105 $D 1 &
launch ep2_fsdp2 1,2 49106 $D 2 $E 2 &
wait

echo "### 8-GPU legs (serial) ###"
ALL=0,1,2,3,4,5,6,7
launch fsdp2_tp2_pp2     $ALL 49111 $PPB $D 2 $T 2 $P 2
launch fsdp2_tp2_cp2     $ALL 49112 $D 2 $T 2 $C 2
launch tp2_pp2_cp2       $ALL 49113 $PPB $D 1 $T 2 $P 2 $C 2
launch fsdp2_pp2_cp2     $ALL 49114 $PPB $D 2 $P 2 $C 2
launch ep2_fsdp2_tp2_pp2 $ALL 49115 $PPB $D 2 $E 2 $T 2 $P 2
launch ep2_fsdp2_tp2_cp2 $ALL 49116 $D 2 $E 2 $T 2 $C 2
launch ep2_fsdp2_pp2_cp2 $ALL 49117 $PPB $D 2 $E 2 $P 2 $C 2
echo "### DONE ###"
