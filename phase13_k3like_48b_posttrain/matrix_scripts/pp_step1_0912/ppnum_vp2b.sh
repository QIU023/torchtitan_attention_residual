#!/bin/bash
# pp2 x vp2 cells; Triton caches nested under the campaign dir (the disk watchdog prunes top-level /tmp/triton_* below 40G free).
set -uo pipefail
export VENV=/workspace/venv_bfx9 PYPRE=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/rankcache:/tmp/attn_gym_up SEED_CFG=kimi_k3_debugmodel
export SEED_ROOT=/workspace/.mx3_seeds_pp100 MEASURE_STEPS=1 WARM_STEPS=1
D=/workspace/ppnum_0912; A=$(ls -dt /workspace/mx3_ppnum_dp1_0912_* | head -1); export INDUCTOR_SEED_CACHE=$A/inductor
B="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
F="--parallelism.data_parallel_shard_degree 1"; P="$F --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
run(){ local tree=$1 tag=$2 gpus=$3 cfg=$4 cell=$5; mkdir -p $D/$tag; rm -f $D/$tag/g.*
  TITAN=$tree CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_BASE=/tmp/triton_ppnum3_$tag GRAD_DUMP=$D/$tag/g CFG=$cfg BATCH="$B"   CELLS="$cell" bash /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh ppnum3_$tag > $D/$tag.mx3.out 2>&1; }
run /tmp/wt_ppnum vp2c 0,1 kimi_k3_debugmodel          "vp2c|2|$P" &
run /tmp/wt_ppnum vp2n 2,3 kimi_k3_debugmodel_pp_naive "vp2n|2|$P" &
wait; df -BG --output=avail /workspace | tail -1
MUTATE=drop_deposit run /tmp/wt_ppmut vp2mut 0,1 kimi_k3_debugmodel "vp2mut|2|$P" &
NOSYNC_GA=1 run /tmp/wt_ppmut dp1ns 2 kimi_k3_debugmodel "dp1ns|1|$F" &
wait
for t in vp2c vp2n vp2mut dp1ns; do R=$(ls -dt /workspace/mx3_ppnum3_${t}_0912_* | head -1); echo "$t: $(grep -v '^seed\|^tree\|^DONE' $R/results.txt)"; ls $D/$t; done
echo VP2-DONE
