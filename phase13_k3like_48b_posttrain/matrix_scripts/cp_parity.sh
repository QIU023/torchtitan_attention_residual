#!/usr/bin/env bash
# CP correctness, not just liveness: context parallel is a pure parallelization of
# the sequence, so cp2 must reproduce the single-rank loss on the same global batch.
# The fsdp2 control in the gate shards the BATCH instead, so it is not this check.
set -uo pipefail
export PYTHONPATH=/workspace/tt_step3
source /venv/main/bin/activate
run() {
  local name=$1 gpus=$2 n=$3; shift 3
  CUDA_VISIBLE_DEVICES=$gpus timeout 1800 torchrun --nproc_per_node=$n \
    --master_port=53101 -m torchtitan.train \
    --module kimi_k3_up --config kimi_k3_up_mini_full_attn \
    --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 5 \
    --training.global-batch-size 4 "$@" \
    --dump-folder /workspace/mx_cppar_$name > /workspace/mx_cppar_$name.log 2>&1
  local rc=$? l
  l=$(grep -oE "loss: +[0-9.]+" /workspace/mx_cppar_$name.log | awk '{print $2}' | uniq | tr '\n' ' ')
  [ -n "$l" ] && echo "$name: $l" || echo "$name: rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' /workspace/mx_cppar_$name.log | grep -oiE '(RuntimeError|ValueError|AssertionError): .{0,110}' | sort -u | head -1)"
}
echo "########## CP parity: single rank vs cp2, same global batch ##########"
run dp1 0 1 --parallelism.data_parallel_shard_degree 1
run cp2 0,1 2 --parallelism.data_parallel_shard_degree 1 --parallelism.context_parallel_degree 2
echo "=== CP PARITY DONE ==="
