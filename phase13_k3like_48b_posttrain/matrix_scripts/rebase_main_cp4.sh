#!/bin/bash
# cp_review4 (K3 CP on fegin's stack): base cells under spmd_types, the packed MLA kernels
# against the upstream generic ones (same seed, same batch), and tp2 x cp2.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"
S="--parallelism.spmd_backend spmd_types"; NOSP="--parallelism.no-enable-sequence-parallel"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_cprun4 CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1 $S
tp2|2|$D 1 $T 2 $S" $MX cp4_base
for cfg in cp2 cp2_generic cp2_allgather cp2_allgather_generic; do
TITAN=/tmp/wt_cprun4 CFG=kimi_k3_debugmodel_$cfg BATCH="$B" CELLS="$cfg|2|$D 1" $MX cp4_$cfg
done
TITAN=/tmp/wt_cprun4 CFG=kimi_k3_debugmodel_cp2 BATCH="$B" \
CELLS="tp2cp2_nosp|4|$D 1 $T 2 $NOSP" $MX cp4_tp2cp2
echo "CP4 MATRICES DONE"
