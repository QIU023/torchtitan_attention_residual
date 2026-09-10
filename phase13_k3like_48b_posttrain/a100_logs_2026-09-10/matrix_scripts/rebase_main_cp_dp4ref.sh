#!/bin/bash
# The dp4 reference the CP table's dp4 x cp2 group lacks (same samples as those rows: dp rank shards).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up
export CUDA_VISIBLE_DEVICES=1,2,3,4 TRITON_CACHE_DIR=/workspace/.triton_cp_dp4
D="--parallelism.data_parallel_shard_degree"; S="--parallelism.spmd_backend spmd_types"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_cprobe CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp4|4|$D 4 $S" $MX cp_dp4ref
echo "CP DP4 REF DONE"
