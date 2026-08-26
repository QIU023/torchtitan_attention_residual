#!/bin/bash
# Do the axes compose? One mesh carrying all of them, with EP on and off.
#
#   pp2 x vp2  x  cp2  x  fsdp2   [x ep2]      = 8 GPUs
#
# The three PRs each measured one axis against pure data parallelism. This asks
# what none of them can: whether a mesh that turns them on TOGETHER still trains
# the same model. It runs on the integration branch, where all three live at
# once.
#
# vp2 rather than plain pp2 on purpose: at vp > 1 each rank holds several
# virtual stages, which is the only shape where the cross-stage delta transport
# engages. Plain 1F1B gives one stage per rank and the transport has nothing to
# reuse -- so a pp2 cell would leave the PP PR's centrepiece untested inside the
# combination.
#
# One table, one run, one seed, and the lattice is CLOSED: every legal subset of
# the four axes at degree 2, ordered by how many are on. EP shards experts inside
# the data axis, so a subset containing EP must contain FSDP -- which is why
# there are twelve rows and not sixteen, and why every EP row sits next to the
# same mesh without it.
#
# Pairs earn their place by being diagnostic: if the three-axis row misbehaves,
# the pairs say WHICH two axes interact. Without them that row is one
# undivided fact.
#
# 32-layer text flavor: layers-per-stage 8 gives 4 stages = pp2 x vp2, evenly,
# which Interleaved1F1B requires. seq 512 satisfies FlexAttention's
# Q_LEN % (cp * 128) == 0 for cp2 and keeps every cell on one curve.
set -uo pipefail
MX=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/mx3.sh
D="--parallelism.data_parallel_shard_degree"
C="--parallelism.context_parallel_degree"
P="--parallelism.pipeline_parallel_degree"
E="--parallelism.expert_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
NB="--parallelism.context_parallel_load_balancer None"
MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
LESS="--parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0"
VP2="$P 2 $L 8 $IL $LESS"
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 512 --training.max-context-length 512 --comm.init-timeout-seconds 3600"

CELLS="dp1|1|$D 1
dp2|2|$D 2
pp2_vp2|2|$D 1 $VP2
cp2|2|$D 1 $C 2 $NB
dp2_ep2|2|$D 2 $E 2
pp2vp2_cp2|4|$D 1 $C 2 $NB $VP2
pp2vp2_fsdp2|4|$D 2 $VP2
cp2_fsdp2|4|$D 2 $C 2 $NB
pp2vp2_fsdp2_ep2|4|$D 2 $VP2 $E 2
cp2_fsdp2_ep2|4|$D 2 $C 2 $NB $E 2
pp2vp2_cp2_fsdp2|8|$D 2 $C 2 $NB $VP2
pp2vp2_cp2_fsdp2_ep2|8|$D 2 $C 2 $NB $VP2 $E 2"

TITAN=/workspace/tt_depfix CFG=kimi_k3_debugmodel_text_32l BATCH="$B" CELLS="$CELLS" $MX cross_axis
echo "CROSS-AXIS MATRIX DONE"
