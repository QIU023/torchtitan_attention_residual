#!/bin/bash
# P7: matched-accumulation dp1 and pp2, flex pinned, activation checkpointing off; from /tmp/wt_ppmut.
set -uo pipefail
until grep -q FLEX-DONE /workspace/ppnum_0912.flex.log 2>/dev/null; do sleep 10; done
export VENV=/workspace/venv_bfx9 PYPRE=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/rankcache:/tmp/attn_gym_up SEED_CFG=kimi_k3_debugmodel TITAN=/tmp/wt_ppmut
export SEED_ROOT=/workspace/.mx3_seeds_pp100 MEASURE_STEPS=1 WARM_STEPS=1 FLEX_NOAUTOTUNE=1
D=/workspace/ppnum_0912; df -BG --output=avail / | tail -1
B="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
F="--parallelism.data_parallel_shard_degree 1"; P="$F --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
run(){ local tag=$1 gpus=$2 cfg=$3 cell=$4; mkdir -p $D/$tag; rm -f $D/$tag/g.*
  CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_BASE=$D/triton/$tag GRAD_DUMP=$D/$tag/g CFG=$cfg BATCH="$B"   CELLS="$cell" bash /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh ppnum6_$tag > $D/$tag.mx3.out 2>&1; }
NOSYNC_GA=1 run n_dp1ns 4 kimi_k3_debugmodel_noac "n_dp1ns|1|$F" &
run n_pp2 5,6 kimi_k3_debugmodel_noac "n_pp2|2|$P" &
wait
for t in n_dp1ns n_pp2; do R=$(ls -dt /workspace/mx3_ppnum6_${t}_0912_* | head -1); echo "$t: $(grep -v '^seed\|^tree\|^DONE' $R/results.txt)"; grep -m1 -o "Applied [A-Za-z]* activation checkpointing" $R/${t}_measure.log || echo "$t: no AC applied"; ls $D/$t; done
echo NOAC-DONE
