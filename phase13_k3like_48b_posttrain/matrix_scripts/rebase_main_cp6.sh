#!/bin/bash
# The CP matrix on cp_pr_candidate 3e268b313 (the declarations PR + the stack copy + the CP layer, on
# upstream/main 390e2985b with 4446): the same eleven cells as the cp_review5 matrix.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel
S="--parallelism.spmd_backend spmd_types"; D="--parallelism.data_parallel_shard_degree"; C="--parallelism.context_parallel_degree"; E="--parallelism.expert_parallel_degree"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_cprun6 CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp1|1|$D 1 $S
dp2|2|$D 2 $S" $MX cp6_base
for cfg in cp2 cp2_generic cp2_allgather cp2_allgather_generic; do
TITAN=/tmp/wt_cprun6 CFG=kimi_k3_debugmodel_$cfg BATCH="$B" CELLS="$cfg|2|$D 1" $MX cp6_$cfg
done
TITAN=/tmp/wt_cprun6 CFG=kimi_k3_debugmodel_cp2 BATCH="$B" CELLS="cp4|4|$D 1 $C 4
cp8|8|$D 1 $C 8
dp2_cp2|4|$D 2
dp2_ep2_cp2|4|$D 2 $E 2" $MX cp6_mix
echo "CP6 MATRIX DONE"
