#!/bin/bash
# The quantile-balancing cells of the QB matrix (the sign-step control rows are in the log already).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_qbrun2 CFG=kimi_k3_debugmodel_qb BATCH="$B" CELLS="dp1|1|$D 1
dp2|2|$D 2
dp2_ep2|2|$D 2 $E 2" $MX qb2b_kimi_k3_debugmodel_qb
echo "QB2 REST DONE"
