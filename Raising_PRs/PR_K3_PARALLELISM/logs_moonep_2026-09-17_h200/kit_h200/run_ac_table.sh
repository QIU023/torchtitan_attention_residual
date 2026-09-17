#!/bin/bash
# AC-reuse table: main vs PR on the K3 debug model, dp1, one GPU, three AC modes, one shared warm inductor cache; floor row = main/none on a fresh cache.
source /workspace/kit/h200/env.sh
OUT=/workspace/ac_$(date +%m%d_%H%M%S); mkdir -p $OUT; R=$OUT/results.txt; echo "main=$(git -C /workspace/tt_main rev-parse --short HEAD) pr=$(git -C /workspace/tt_ac rev-parse --short HEAD)" > $R
B="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 512 --debug.seed 42 --debug.deterministic --parallelism.data_parallel_shard_degree 1 --metrics.log_freq 1"
SHARED=$OUT/cache_shared
run(){ local nm=$1 tree=$2 cache=$3 steps=$4; shift 4; local d=$OUT/$nm; mkdir -p $d
  ( cd $tree && CUDA_VISIBLE_DEVICES=${GPU:-0} TRITON_CACHE_DIR=$cache/triton TORCHINDUCTOR_CACHE_DIR=$cache/inductor PYTHONPATH=$tree timeout 1800 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --training.steps $steps $B --dump-folder $d "$@" > $d/log.txt 2>&1 ); local rc=$?
  sed "s/\x1b\[[0-9;]*m//g" $d/log.txt | grep -oE "step: *[0-9]+ +loss: *[0-9.]+ +grad_norm: *[0-9.]+ +memory: *[0-9.]+GiB\([0-9.]+%\) +tps: *[0-9,]+" | tr -s " " > $d/steps.txt
  echo "$nm rc=$rc steps=$(wc -l < $d/steps.txt) | $(grep -E "^step: 1 " $d/steps.txt | head -1) | $(grep -E "^step: 10 " $d/steps.txt | head -1)" >> $R; tail -1 $R
  grep -a -m1 "Error\|error:" $d/log.txt | grep -v lspci | cut -c1-160 >> $R; }
declare -A TREE=([main]=/workspace/tt_main [pr]=/workspace/tt_ac)
declare -A ACF=([none]="activation-checkpoint:none" [selective]="" [full]="activation-checkpoint:full")
for t in main pr; do for m in none selective full; do run warm_${t}_${m} ${TREE[$t]} $SHARED 1 ${ACF[$m]}; done; done
for t in main pr; do for m in none selective full; do run ${t}_${m} ${TREE[$t]} $SHARED 10 ${ACF[$m]}; done; done
run warm_floor_main_none ${TREE[main]} $OUT/cache_fresh 1 ${ACF[none]}; run floor_main_none ${TREE[main]} $OUT/cache_fresh 10 ${ACF[none]}
echo "DONE $OUT" >> $R
