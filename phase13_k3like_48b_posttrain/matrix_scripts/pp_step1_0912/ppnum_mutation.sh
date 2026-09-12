#!/bin/bash
# Mutation control: pp2 x vp2 cached with one real bug injected (a rank's deposit is never added back).
set -uo pipefail
export VENV=/workspace/venv_bfx9 PYPRE=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/rankcache:/tmp/attn_gym_up TITAN=/tmp/wt_ppmut SEED_CFG=kimi_k3_debugmodel
export SEED_ROOT=/workspace/.mx3_seeds_pp100 MEASURE_STEPS=1 WARM_STEPS=1
D=/workspace/ppnum_0912; A=$(ls -dt /workspace/mx3_ppnum_dp1_0912_* | head -1)
export INDUCTOR_SEED_CACHE=$A/inductor
B="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
P="--parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
mkdir -p $D/vp2mut $D/dp1ns
MUTATE=drop_deposit CUDA_VISIBLE_DEVICES=1,2 TRITON_CACHE_BASE=/tmp/triton_ppnum_mut GRAD_DUMP=$D/vp2mut/g CFG=kimi_k3_debugmodel BATCH="$B"   CELLS="vp2mut|2|$P" bash /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh ppnum_mut > $D/vp2mut.mx3.out 2>&1 &
NOSYNC_GA=1 CUDA_VISIBLE_DEVICES=0 TRITON_CACHE_BASE=/tmp/triton_ppnum_nosync GRAD_DUMP=$D/dp1ns/g CFG=kimi_k3_debugmodel BATCH="$B" \
  CELLS="dp1ns|1|--parallelism.data_parallel_shard_degree 1" bash /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh ppnum_dp1ns > $D/dp1ns.mx3.out 2>&1 &
wait
R=$(ls -dt /workspace/mx3_ppnum_mut_0912_* | head -1); cat $R/results.txt; ls $D/vp2mut; echo MUT-DONE
