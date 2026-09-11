#!/bin/bash
set -uo pipefail
MX=/workspace/mx3.sh
export VENV=/venv/main PYPRE=/workspace/attn_gym_up TITAN=/workspace/titan
export SEED_ROOT=/workspace/.mx3_seeds_pp100deep SEED_CFG=kimi_k3_debugmodel_deep
export MEASURE_STEPS=100 WARM_STEPS=1
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
MB="--parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"

TRITON_CACHE_DIR=/tmp/triton_deep_a CFG=kimi_k3_debugmodel_deep BATCH="$B" \
CELLS="dp1|1|$D 1
pp2|2|$D 1 $P 2 $MB
pp2vp2|2|$D 1 $P 2 $MB $IL" $MX deep_a

TRITON_CACHE_DIR=/tmp/triton_deep_b CFG=kimi_k3_debugmodel_deep_ppnaive BATCH="$B" \
CELLS="pp2vp2n|2|$D 1 $P 2 $MB $IL" $MX deep_b

MB_REVERSE=1 TRITON_CACHE_DIR=/tmp/triton_deep_c CFG=kimi_k3_debugmodel_deep BATCH="$B" \
CELLS="dp1rev|1|$D 1" $MX deep_c
echo DEEP-DONE
touch /workspace/deep.done
