#!/bin/bash
# PP on the rebased pp_review3 (0e7cc5ea1 on upstream/main 6e2ac3dcd): the bf16 grad-norm matrix on
# the branch as it is (wt_pprun3), then the float32 grad-norm matrix with PR 4135's reduction applied
# to the run tree only (wt_pprun3gn). Same seed, cells and batch as the d6b1ffe47 matrices.
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main30
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
LESS="--parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
for pair in "wt_pprun3 pp3" "wt_pprun3gn pp3gn"; do
set -- $pair
TITAN=/tmp/$1 CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1
pp2_vp4|2|$D 1 $P 2 $L 4 $IL
pp4_vp4|4|$D 1 $P 4 $L 2 $IL
pp8_vp4|8|$D 1 $P 8 $L 1 $IL" $MX ${2}_pp
TITAN=/tmp/$1 CFG=kimi_k3_debugmodel_pp_naive BATCH="$B" \
CELLS="pp8_vp4n|8|$D 1 $P 8 $L 1 $IL" $MX ${2}_ppn
done
# the even split row and the data-parallel / expert-parallel meshes around PP (bf16 only)
E="--parallelism.expert_parallel_degree"
TITAN=/tmp/wt_pprun3 CFG=kimi_k3_debugmodel BATCH="$B" \
CELLS="pp2_vp4_even|2|$D 1 $P 2 $L 4 $IL $LESS
dp2|2|$D 2
dp2_ep2|2|$D 2 $E 2
dp2_pp2_vp4|4|$D 2 $P 2 $L 4 $IL
dp2_ep2_pp2_vp4|4|$D 2 $E 2 $P 2 $L 4 $IL
dp2_pp4_vp4|8|$D 2 $P 4 $L 2 $IL
dp2_ep2_pp4_vp4|8|$D 2 $E 2 $P 4 $L 2 $IL" $MX pp3_pp_mesh
echo "PP3 MATRICES DONE"
