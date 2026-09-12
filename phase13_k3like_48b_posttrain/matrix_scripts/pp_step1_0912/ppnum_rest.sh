#!/bin/bash
# Remaining step-1 cells on dp1's warm inductor cache: dp1b + pp2, then vp2 cached + vp2 naive.
set -uo pipefail
export VENV=/workspace/venv_bfx9 PYPRE=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/rankcache:/tmp/attn_gym_up TITAN=/tmp/wt_ppnum SEED_CFG=kimi_k3_debugmodel
export SEED_ROOT=/workspace/.mx3_seeds_pp100 MEASURE_STEPS=1 WARM_STEPS=1
D=/workspace/ppnum_0912
A=$(ls -dt /workspace/mx3_ppnum_dp1_0912_* | head -1); echo "dp1 run: $A"
until grep -q '^DONE' $A/results.txt 2>/dev/null; do sleep 10; done; cat $A/results.txt
test -d $A/inductor || { echo "no warm cache"; exit 1; }
export INDUCTOR_SEED_CACHE=$A/inductor
B="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
F="--parallelism.data_parallel_shard_degree 1"; P="$F --parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
run(){ local tag=$1 gpus=$2 cfg=$3 cell=$4; mkdir -p $D/$tag
  CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_BASE=/tmp/triton_ppnum_$tag GRAD_DUMP=$D/$tag/g CFG=$cfg BATCH="$B"   CELLS="$cell" bash /workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh ppnum_$tag > $D/$tag.mx3.out 2>&1; }
run dp1b 0   kimi_k3_debugmodel          "dp1b|1|$F" &
run pp2  1,2 kimi_k3_debugmodel          "pp2|2|$P" &
wait; df -BG --output=avail /workspace | tail -1
run vp2c 3,4 kimi_k3_debugmodel          "vp2c|2|$P $IL" &
run vp2n 5,6 kimi_k3_debugmodel_pp_naive "vp2n|2|$P $IL" &
wait
for t in dp1b pp2 vp2c vp2n; do R=$(ls -dt /workspace/mx3_ppnum_${t}_0912_* | head -1); echo "$R"; grep -v '^seed\|^tree' $R/results.txt; done
ls -la $D/*/; echo PPNUM-DONE
