#!/bin/bash
# QB on qb_release a4658eefe (upstream/main 390e2985b, 4446 merged: the flavor no longer pins a backend, and under the default spmd_types the multimodal inputs have no layout on main, so every cell names partial_dtensor): sign-step control and quantile balancing,
# dp1 / dp2 / dp2 x ep2 -- the ep cell is where the balancing is exercised.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
PD="--parallelism.spmd_backend partial_dtensor"
for cfg in kimi_k3_debugmodel kimi_k3_debugmodel_qb; do
TITAN=/tmp/wt_qbrun3 CFG=$cfg BATCH="$B" CELLS="dp1|1|$D 1 $PD
dp2|2|$D 2 $PD
dp2_ep2|2|$D 2 $E 2 $PD" $MX qb3b_$cfg
done
echo "QB3B MATRIX DONE"
