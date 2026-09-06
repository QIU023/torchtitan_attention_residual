#!/bin/bash
# cp_review5 without the parallelize CP hook (61a73ca6c): the splice takes the cp group from the SPMD mesh.
# Same group, same arithmetic: cp2 and dp2 x cp2 must read the CP matrix's step 1 (12.53972, 12.52908).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up TITAN=/tmp/wt_cprobe
export CUDA_VISIBLE_DEVICES=4,5,6,7 TRITON_CACHE_DIR=/workspace/.triton_cp8_hookless
D="--parallelism.data_parallel_shard_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
CFG=kimi_k3_debugmodel_cp2 BATCH="$B" CELLS="cp2|2|$D 1
dp2_cp2|4|$D 2" $MX cp8_hookless
echo "CP8 HOOKLESS DONE"
