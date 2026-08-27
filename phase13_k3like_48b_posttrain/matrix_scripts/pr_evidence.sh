#!/bin/bash
# All PR evidence matrices, 10 steps, reported at steps 1 / 3 / 10.
# Each matrix runs on the worktree of the branch that will be filed, not on the
# integration tree, so the numbers belong to the diff a reviewer sees.
set -uo pipefail
MX=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mx3.sh
chmod +x $MX

D="--parallelism.data_parallel_shard_degree"
P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
# num-pp-microbatches defaults to 1, which is fewer than the stage count and
# raises before the first step; every PP cell has to set it.
MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
LESS="--parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0"
C="--parallelism.context_parallel_degree"
NB="--parallelism.context_parallel_load_balancer None"
E="--parallelism.expert_parallel_degree"

# ---- PP: pp x vp cross product, 32-layer text flavor, dp=1 --------------------
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
pp8_vp4|8|$D 1 $P 8 $L 1 $IL $LESS" $MX ppvp

# ---- PP: 2D mesh with data parallel ------------------------------------------
TITAN=/workspace/tt_pptext CFG=kimi_k3_debugmodel \
BATCH="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256" \
CELLS="dp2|2|$D 2
fsdp2_pp2|4|$D 2 $P 2 $MB $LESS
fsdp2_pp4|8|$D 2 $P 4 $MB $LESS" $MX ppdp

# ---- CP: dp1 vs cp2/cp4/cp8 at seq 1024 --------------------------------------
TITAN=/workspace/tt_cptext CFG=kimi_k3_debugmodel \
BATCH="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 1024 --training.max-context-length 1024" \
CELLS="dp1|1|$D 1
cp2|2|$D 1 $C 2 $NB
cp4|4|$D 1 $C 4 $NB
cp8|8|$D 1 $C 8 $NB" $MX cpseq

# ---- CP: 2D mesh with data parallel, seq 512 so cp2 and cp4 share a baseline --
TITAN=/workspace/tt_cptext CFG=kimi_k3_debugmodel \
BATCH="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 512 --training.max-context-length 512" \
CELLS="dp2|2|$D 2
fsdp2_cp2|4|$D 2 $C 2 $NB
fsdp2_cp4|8|$D 2 $C 4 $NB" $MX cpdp

# ---- EP: ep2/ep4/ep8 each against the same-world-size pure dp baseline --------
TITAN=/workspace/tt_ep CFG=kimi_k3_debugmodel \
BATCH="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256" \
CELLS="dp2|2|$D 2
ep2_fsdp2|2|$D 2 $E 2
dp4|4|$D 4
ep4_fsdp4|4|$D 4 $E 4
dp8|8|$D 8
ep8_fsdp8|8|$D 8 $E 8" $MX epdp

echo "ALL MATRICES DONE"
