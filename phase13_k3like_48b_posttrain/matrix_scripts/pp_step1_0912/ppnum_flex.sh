#!/bin/bash
# P6: every cell with flex compiled without autotune (FLEX_NOAUTOTUNE=1), from /tmp/wt_ppmut.
set -uo pipefail
export VENV=/workspace/venv_bfx9 PYPRE=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/rankcache:/tmp/attn_gym_up SEED_CFG=kimi_k3_debugmodel TITAN=/tmp/wt_ppmut
export SEED_ROOT=/workspace/.mx3_seeds_pp100 MEASURE_STEPS=1 WARM_STEPS=1 FLEX_NOAUTOTUNE=1
D=/workspace/ppnum_0912; df -BG --output=avail / | tail -1
B="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
F="--parallelism.data_parallel_shard_degree 1"; P="$F --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
run(){ local tag=$1 gpus=$2 cfg=$3 cell=$4; mkdir -p $D/$tag; rm -f $D/$tag/g.*
  CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_BASE=$D/triton/$tag GRAD_DUMP=$D/$tag/g CFG=$cfg BATCH="$B"   CELLS="$cell" bash /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh ppnum5_$tag > $D/$tag.mx3.out 2>&1; }
NOSYNC_GA=1 run f_dp1ns 0 kimi_k3_debugmodel "f_dp1ns|1|$F" &
run f_pp2 1,2 kimi_k3_debugmodel "f_pp2|2|$P" &
wait; df -BG --output=avail / | tail -1
run f_vp2n 0,1 kimi_k3_debugmodel_pp_naive "f_vp2n|2|$P $IL" &
run f_vp2c 2,3 kimi_k3_debugmodel "f_vp2c|2|$P $IL" &
wait
for t in f_dp1ns f_pp2 f_vp2n f_vp2c; do R=$(ls -dt /workspace/mx3_ppnum5_${t}_0912_* | head -1); echo "$t: $(grep -v '^seed\|^tree\|^DONE' $R/results.txt)"; ls $D/$t; done
echo FLEX-DONE
