#!/bin/bash
# AC-reuse memory sweep, activation checkpointing off, main vs PR: tokens per micro-batch x depth, 3 steps each,
# four runs at a time on the four GPUs (each its own cache; memory and loss are per GPU, tps shares the host).
source /workspace/kit/h200/env.sh
OUT=/workspace/acsc_$(date +%m%d_%H%M%S); mkdir -p $OUT; R=$OUT/results.txt; echo "main=$(git -C /workspace/tt_main rev-parse --short HEAD) pr=$(git -C /workspace/tt_ac rev-parse --short HEAD)" > $R
declare -A TREE=([main]=/workspace/tt_main [pr]=/workspace/tt_ac)
run(){ local nm=$1 tree=$2 gpu=$3 cfg=$4 tok=$5; local d=$OUT/$nm; mkdir -p $d
  ( cd $tree && CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=$d/triton TORCHINDUCTOR_CACHE_DIR=$d/inductor PYTHONPATH=$tree timeout 2400 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train --module kimi_k3 --config $cfg --training.steps 3 --training.num-tokens-per-train-step $tok --training.num-tokens-per-microbatch-per-dp-rank $tok --debug.seed 42 --debug.deterministic --parallelism.data_parallel_shard_degree 1 --metrics.log_freq 1 --dump-folder $d activation-checkpoint:none > $d/log.txt 2>&1 ); local rc=$?
  sed "s/\x1b\[[0-9;]*m//g" $d/log.txt | grep -oE "step: *[0-9]+ +loss: *[0-9.]+ +grad_norm: *[0-9.]+ +memory: *[0-9.]+GiB\([0-9.]+%\) +tps: *[0-9,]+" | tr -s " " > $d/steps.txt
  local oom=$(grep -a -c "OutOfMemoryError\|CUDA out of memory" $d/log.txt)
  echo "$nm rc=$rc oom=$oom | $(grep -E "^step: 3 " $d/steps.txt | head -1)" >> $R
  rm -rf $d/triton $d/inductor; }
wave(){ local i=0; for spec in "$@"; do IFS=: read nm t cfg tok <<< "$spec"; run $nm ${TREE[$t]} $i $cfg $tok & i=$((i+1)); done; wait; sort $R -o $R.tmp; }
wave main_24l_512:main:kimi_k3_debugmodel:512 pr_24l_512:pr:kimi_k3_debugmodel:512 main_24l_4096:main:kimi_k3_debugmodel:4096 pr_24l_4096:pr:kimi_k3_debugmodel:4096
wave main_24l_8192:main:kimi_k3_debugmodel:8192 pr_24l_8192:pr:kimi_k3_debugmodel:8192 main_48l_4096:main:kimi_k3_debugmodel_48l:4096 pr_48l_4096:pr:kimi_k3_debugmodel_48l:4096
wave main_48l_8192:main:kimi_k3_debugmodel_48l:8192 pr_48l_8192:pr:kimi_k3_debugmodel_48l:8192 main_24l_16384:main:kimi_k3_debugmodel:16384 pr_24l_16384:pr:kimi_k3_debugmodel:16384
wave main_48l_16384:main:kimi_k3_debugmodel_48l:16384 pr_48l_16384:pr:kimi_k3_debugmodel_48l:16384 main_24l_b24_4096:main:kimi_k3_debugmodel_b24:4096 pr_24l_b24_4096:pr:kimi_k3_debugmodel_b24:4096
echo "DONE $OUT" >> $R
