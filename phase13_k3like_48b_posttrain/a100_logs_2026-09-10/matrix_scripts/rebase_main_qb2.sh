#!/bin/bash
# QB on qb_release rebased onto upstream/main 6e2ac3dcd: sign-step control and quantile balancing,
# dp1 / dp2 / dp2 x ep2 -- the ep cell is where the balancing is exercised.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
for cfg in kimi_k3_debugmodel kimi_k3_debugmodel_qb; do
TITAN=/tmp/wt_qbrun2 CFG=$cfg BATCH="$B" CELLS="dp1|1|$D 1
dp2|2|$D 2
dp2_ep2|2|$D 2 $E 2" $MX qb2_$cfg
done
echo "QB2 MATRIX DONE"
