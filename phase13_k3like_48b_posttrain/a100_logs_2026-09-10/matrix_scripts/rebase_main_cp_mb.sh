#!/bin/bash
# Under context parallel the trainer packs each dp rank's micro-batch 2 x cp times longer
# (trainer.py: num_tokens_per_pp_microbatch * (2 * cp)), so the cp2 cells train on 1024-token
# micro-batches and cp8 on 4096, while dp1 packs 256. Same documents, other packing. These dp1 cells
# use the CP cells' micro-batch so the step-1 comparison is on the same packing.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up
D="--parallelism.data_parallel_shard_degree"; S="--parallelism.spmd_backend spmd_types"
mb=$1; gpu=$2
export CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=/workspace/.triton_cp_mb$mb
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank $mb"
TITAN=/tmp/wt_cprobe CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp1_mb$mb|1|$D 1 $S" $MX cp_mb$mb
echo "CP MB$mb DONE"
