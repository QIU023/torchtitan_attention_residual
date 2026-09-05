#!/bin/bash
# ac_reuse_attention on ac_review2 (upstream/main 6e2ac3dcd): the flavor as it is (selective AC over
# the whole block) against the alias that sets the flag, on dp1 / dp2 / dp2 x ep2. Expected bitwise.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
for cfg in kimi_k3_debugmodel kimi_k3_debugmodel_ac_reuse; do
TITAN=/tmp/wt_acrun CFG=$cfg BATCH="$B" CELLS="dp1|1|$D 1 $PD
dp2|2|$D 2 $PD
dp2_ep2|2|$D 2 $E 2 $PD" $MX ac3_$cfg
done
echo "AC MATRIX DONE"
