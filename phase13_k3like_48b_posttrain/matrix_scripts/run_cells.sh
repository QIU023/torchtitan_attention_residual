#!/usr/bin/env bash
# Run named cells from the 13-cell matrix, with their arguments taken FROM
# run13_flav.sh rather than retyped.
#
#   TITAN=/workspace/tt_x OUT=/workspace/mx_x FLAVOR=... bash run_cells.sh tp2 ep2_fsdp2_tp2_cp2
#
# Six 8-GPU runs were once spent chasing a "deadlock" that was a hand-written
# --training.local-batch-size 2, a flag the matrix passes only to PP cells. So this
# script never spells a cell's arguments: it greps the `launch <name> ...` line out of
# run13_flav.sh, expands the same variables that file defines, and runs that.
#
# Cells are SERIAL here regardless of which GPU set the matrix gives them. This is for
# targeted re-checks after a change, where the point is a clean comparison, not throughput.
set -uo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
TITAN=${TITAN:?set TITAN to the tree under test}
OUT=${OUT:-/workspace/mx_cells}
STEPS=${STEPS:-10}
EXTRA=${EXTRA:-}
FLAVOR=${FLAVOR:-kimi_k3_debugmodel_report_arch}
mkdir -p "$OUT"; cd "$TITAN"; export PYTHONPATH=$TITAN
source /venv/main/bin/activate

BASE="--module kimi_k3 --config $FLAVOR --debug.seed 42 \
 --debug.deterministic --metrics.log_freq 1 --training.steps $STEPS \
 --training.global-batch-size 8 $EXTRA"
PPB="--training.local-batch-size 2"
D=--parallelism.data_parallel_shard_degree
T=--parallelism.tensor_parallel_degree
P=--parallelism.pipeline_parallel_degree
C=--parallelism.context_parallel_degree
E=--parallelism.expert_parallel_degree
ALL=0,1,2,3,4,5,6,7

echo "=== TITAN=$TITAN FLAVOR=$FLAVOR OUT=$OUT ==="

for name in "$@"; do
  line=$(grep -E "^launch +${name} " "$HERE/run13_flav.sh" | head -1)
  if [ -z "$line" ]; then
    echo "$name: no such cell in run13_flav.sh" >&2
    continue
  fi
  # launch <name> <gpus> <port> <args...>
  rest=$(sed -E "s/^launch +${name} +//" <<<"$line" | sed 's/ *&$//')
  gpus=$(awk '{print $1}' <<<"$rest")
  port=$(awk '{print $2}' <<<"$rest")
  cellargs=$(cut -d' ' -f3- <<<"$rest")
  eval "gpus=\"$gpus\"; cellargs=\"$cellargs\""
  n=$(awk -F, '{print NF}' <<<"$gpus")

  echo "--- $name  gpus=$gpus  args: $cellargs"
  # Retry on a fresh port. A socket left in TIME_WAIT by an earlier run makes torchrun
  # die with EADDRINUSE, which reads exactly like a numerical failure -- it cost this
  # script one cell the first time it ran. run13_flav.sh retries for the same reason.
  for attempt in 1 2 3; do
    use_port=$((port + 900 + 400 * (attempt - 1)))
    rm -rf "$OUT/$name"
    CUDA_VISIBLE_DEVICES="$gpus" timeout 7200 torchrun \
      --nproc_per_node="$n" --master_port="$use_port" ${ENTRY:--m torchtitan.train} \
      $BASE $cellargs --dump-folder "$OUT/$name" > "$OUT/$name.log" 2>&1
    rc=$?
    grep -q EADDRINUSE "$OUT/$name.log" || break
    echo "  ($name: port $use_port in use, retrying)"
  done
  losses=$(grep -oE "loss: +[0-9.]+" "$OUT/$name.log" | awk '{printf "%s ", $2}')
  if [ -n "$losses" ]; then
    echo "    $losses"
  else
    # Match any *Error:, not an enumerated list. A whitelist silently produces
    # "rc=N FAIL:" with no reason whenever the failure is a type nobody listed --
    # a FileNotFoundError from a relative tokenizer path did exactly that.
    echo "    rc=$rc FAIL: $(sed 's/\x1b\[[0-9;]*m//g' "$OUT/$name.log" \
      | grep -oE '[A-Za-z_]*Error: .{0,110}' | sort -u | head -2 | tr '\n' ' ')"
  fi
  rm -rf "$OUT/$name/checkpoint"
done
