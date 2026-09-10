#!/bin/bash
# PP and CP evidence matrices with PR-4135 (fp32 grad-norm reduction) applied.
# EP skipped for this pass. Steps 1 / 3 / 10, one seed per matrix, per-cell warm-up.
set -uo pipefail
MX=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mx3.sh
chmod +x $MX

D="--parallelism.data_parallel_shard_degree"
P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
LESS="--parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0"
C="--parallelism.context_parallel_degree"
NB="--parallelism.context_parallel_load_balancer None"

TITAN=/workspace/tt_pptext CFG=kimi_k3_debugmodel_32l \
BATCH="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256" \
CELLS="dp1|1|$D 1
pp2|2|$D 1 $P 2 $MB $LESS
pp4|4|$D 1 $P 4 $MB $LESS
pp2_vp2|2|$D 1 $P 2 $L 8 $IL $LESS
pp2_vp4|2|$D 1 $P 2 $L 4 $IL $LESS
pp4_vp2|4|$D 1 $P 4 $L 4 $IL $LESS
pp4_vp4|4|$D 1 $P 4 $L 2 $IL $LESS
pp8_vp2|8|$D 1 $P 8 $L 2 $IL $LESS
pp8_vp4|8|$D 1 $P 8 $L 1 $IL $LESS" $MX gn_ppvp

TITAN=/workspace/tt_pptext CFG=kimi_k3_debugmodel \
BATCH="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256" \
CELLS="dp2|2|$D 2
fsdp2_pp2|4|$D 2 $P 2 $MB $LESS
fsdp2_pp4|8|$D 2 $P 4 $MB $LESS" $MX gn_ppdp

TITAN=/workspace/tt_cptext CFG=kimi_k3_debugmodel \
BATCH="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 1024 --training.max-context-length 1024" \
CELLS="dp1|1|$D 1
cp2|2|$D 1 $C 2 $NB
cp4|4|$D 1 $C 4 $NB
cp8|8|$D 1 $C 8 $NB" $MX gn_cpseq

TITAN=/workspace/tt_cptext CFG=kimi_k3_debugmodel \
BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 512 --training.max-context-length 512" \
CELLS="dp2|2|$D 2
fsdp2_cp2|4|$D 2 $C 2 $NB
fsdp2_cp4|8|$D 2 $C 4 $NB" $MX gn_cpdp

echo "PP AND CP MATRICES DONE"
