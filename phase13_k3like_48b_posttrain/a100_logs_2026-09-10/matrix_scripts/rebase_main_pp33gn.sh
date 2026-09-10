#!/bin/bash
# The 33-layer PP matrix with the total gradient norm carried in float32 (PR 4135's reduction on the
# run tree, not on the branch); same seed, batch and cells as the bf16 matrix.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main33 SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"; E="--parallelism.expert_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_pprun33gn CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1
pp2_vp4|2|$D 1 $P 2 $L 4 $IL
pp4_vp4|4|$D 1 $P 4 $L 2 $IL
pp8_vp4|8|$D 1 $P 8 $L 1 $IL
dp2|2|$D 2
dp2_ep2|2|$D 2 $E 2
dp2_pp2_vp4|4|$D 2 $P 2 $L 4 $IL
dp2_ep2_pp2_vp4|4|$D 2 $E 2 $P 2 $L 4 $IL
dp2_pp4_vp4|8|$D 2 $P 4 $L 2 $IL
dp2_ep2_pp4_vp4|8|$D 2 $E 2 $P 4 $L 2 $IL" $MX pp33gn_pp
TITAN=/tmp/wt_pprun33gn CFG=kimi_k3_debugmodel_pp_naive BATCH="$B" \
CELLS="pp8_vp4n|8|$D 1 $P 8 $L 1 $IL" $MX pp33gn_ppn
echo "PP33GN MATRIX DONE"
