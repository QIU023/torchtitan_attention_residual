#!/bin/bash
# 33 layers, step 1 only, full-precision loss for dp1 vs pp2 (printed digits are not enough).
set -uo pipefail
source /venv/main/bin/activate
cd /workspace/titan
export PYTHONPATH=/workspace/attn_gym_up:/workspace/titan LOSS_FULL=1
S=$(ls -d /workspace/.mx3_seeds_pp100deep/kimi_k3_debugmodel_deep_* 2>/dev/null | head -1)
B="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
run(){ nm=$1 np=$2; shift 2
  d=/workspace/fl_$nm; rm -rf $d; mkdir -p $d
  [ -n "$S" ] && cp -r "$S/seed_ckpt" "$d/checkpoint"
  TRITON_CACHE_DIR=/tmp/triton_fl_$nm timeout 1800 torchrun --nproc_per_node=$np --master_port=$((33000+RANDOM%9000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_deep --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps 1 $B --checkpoint.enable --checkpoint.interval 100000 "$@" \
    --dump-folder $d > /workspace/fl_$nm.log 2>&1; echo "$nm rc=$?"; grep -a FULLLOSS /workspace/fl_$nm.log | head -2; }
run dp1 1 --parallelism.data_parallel_shard_degree 1
run pp2 2 --parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4
echo FULLLOSS-DONE; touch /workspace/fullloss.done
