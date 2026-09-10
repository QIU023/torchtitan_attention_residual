#!/bin/bash
# 100-step curves on data that does not repeat: the debug model on the streamed cc12m (the
# 32-sample test set is memorized by step 90, which makes its late window a memorization race).
# dp1, pp2 x vp4, pp8 x vp4 (delta transport) and pp8 x vp4 with the whole stack every hop,
# same seed checkpoint and batch as the matrix; dp=1 everywhere, so every cell reads the same stream.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main30 SEED_CFG=kimi_k3_debugmodel MEASURE_STEPS=100
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_pprun3 CFG=kimi_k3_debugmodel_cc12m BATCH="$B" \
CELLS="dp1|1|$D 1
pp2_vp4|2|$D 1 $P 2 $L 4 $IL
pp8_vp4|8|$D 1 $P 8 $L 1 $IL" $MX pp3c100cc_pp
TITAN=/tmp/wt_pprun3 CFG=kimi_k3_debugmodel_cc12m_pp_naive BATCH="$B" \
CELLS="pp8_vp4n|8|$D 1 $P 8 $L 1 $IL" $MX pp3c100cc_ppn
echo "PP CURVES100 CC12M DONE"
