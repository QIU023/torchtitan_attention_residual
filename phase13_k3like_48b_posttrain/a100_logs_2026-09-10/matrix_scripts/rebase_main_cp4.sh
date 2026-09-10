#!/bin/bash
# The cp2 cells of cp_review4 with the seed built from the plain flavor (same shape, same init).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"
NOSP="--parallelism.no-enable-sequence-parallel"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
for cfg in cp2 cp2_generic cp2_allgather cp2_allgather_generic; do
TITAN=/tmp/wt_cprun4 CFG=kimi_k3_debugmodel_$cfg BATCH="$B" CELLS="$cfg|2|$D 1" $MX cp4r_$cfg
done
TITAN=/tmp/wt_cprun4 CFG=kimi_k3_debugmodel_cp2 BATCH="$B" \
CELLS="tp2cp2_nosp|4|$D 1 $T 2 $NOSP" $MX cp4r_tp2cp2
echo "CP4R DONE"
