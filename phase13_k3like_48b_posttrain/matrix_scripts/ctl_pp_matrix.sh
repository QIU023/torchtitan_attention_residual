#!/bin/bash
# The control the PP numerics argument needs: upstream's own models on upstream's own pipeline
# (pipeline_llm, PipelineStage, the stock schedules), no Kimi K3 code in the run. Pure upstream/main
# 390e2985b (worktree /tmp/wt_main4446; its local hacks touch kimi_k3 only). Same protocol as the
# K3 PP tables: one seed checkpoint per flavor, 10 steps, --debug.deterministic, 4096 tokens per
# step in 256-token micro-batches, partial_dtensor. Both debug flavors have 6 layers (8 units).
set -uo pipefail
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_ctl
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
IL="$MB --parallelism.pipeline_parallel_schedule Interleaved1F1B"
PD="--parallelism.spmd_backend partial_dtensor"
export SEED_EXTRA="$PD"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256"
for m in llama3 deepseek_v3; do
MODULE=$m SEED_CFG=${m}_debugmodel TITAN=/tmp/wt_main4446 CFG=${m}_debugmodel BATCH="$B" \
CELLS="dp1|1|$D 1 $PD
pp2|2|$D 1 $P 2 $MB $PD
pp2_vp4|2|$D 1 $P 2 $L 1 $IL $PD
pp4_vp2|4|$D 1 $P 4 $L 1 $IL $PD
pp8|8|$D 1 $P 8 $MB $PD" $MX ctl_$m
done
echo "CTL PP MATRIX DONE"
