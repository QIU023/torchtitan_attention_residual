#!/bin/bash
# attn_res_cache_offload on pp_offload_review1 (20d83a8cb on the 33-layer pp_review3): pp2 x vp4 and
# pp8 x vp4 with the store on pinned host memory, against the pp3 matrix's on-device rows.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main33 SEED_CFG=kimi_k3_debugmodel
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_ppoffrun33 CFG=kimi_k3_debugmodel_pp_offload BATCH="$B" \
CELLS="pp2_vp4_offload|2|$D 1 $P 2 $L 4 $IL
pp8_vp4_offload|8|$D 1 $P 8 $L 1 $IL" $MX ppoff33_pp
echo "PPOFF33 MATRIX DONE"
