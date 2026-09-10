#!/bin/bash
# The typecheck rows of 4492's table 2 on THE cache of the backend-pair table: this run starts from the
# pairs run's inductor cache (INDUCTOR_SEED_CACHE) and shares its triton cache, gym b19162e (/tmp/attn_gym_up).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel
export CUDA_VISIBLE_DEVICES=4,5 TRITON_CACHE_DIR=/workspace/.triton_decl_pairs_b19 PYPRE=/tmp/attn_gym_up
export INDUCTOR_SEED_CACHE=/workspace/mx3_decl_pairs_b19_0906_195423/inductor
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_declrun CFG=kimi_k3_debugmodel_tc BATCH="$B" \
CELLS="dp1_tc|1|$D 1
dp2_tc|2|$D 2
dp2_ep2_tc|2|$D 2 $E 2" $MX decl_tc_onecache
echo "DECL TC ONECACHE DONE"
