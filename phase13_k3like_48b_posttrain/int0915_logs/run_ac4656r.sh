#!/bin/bash
# AC recompute on main 0ae22167c (KDA guard lifted locally): dp1 with AC none / selective / full, seeded, 3 steps.
set -uo pipefail
T=/tmp/wt_ac_4656; OUT=/workspace/.smoke_int0915/ac4656r_$(date +%H%M%S); mkdir -p $OUT
B="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 512 --debug.seed 42 --debug.deterministic --parallelism.data_parallel_shard_degree 1"
run(){ local nm=$1 gpu=$2; shift 2; local d=$OUT/$nm; mkdir -p $d
  ( source /workspace/venv_bfx9/bin/activate && cd $T && CUDA_VISIBLE_DEVICES=$gpu TRITON_CACHE_DIR=$d/triton TORCHINDUCTOR_CACHE_DIR=$d/inductor \
    PYTHONPATH=/workspace/pylib/attn_gym_main:$T timeout 1800 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --metrics.log_freq 1 --training.steps 3 $B --dump-folder $d "$@" > $d/log.txt 2>&1 )
  echo "$nm rc=$?" >> $OUT/rc.txt
  sed 's/\x1b\[[0-9;]*m//g' $d/log.txt | grep -oE 'step: *[0-9]+ +loss: *[0-9.]+ +grad_norm: *[0-9.]+ +memory: *[0-9.]+GiB' | sort -u | tr -s ' ' >> $OUT/rc.txt
  sed 's/\x1b\[[0-9;]*m//g' $d/log.txt | grep -E "Error" | grep -v lspci | head -2 | cut -c1-200 >> $OUT/rc.txt
  rm -rf $d/triton $d/inductor; }
run ac_none 4 activation-checkpoint:none &
run ac_selective 5 &
run ac_full 6 activation-checkpoint:full &
wait; echo DONE >> $OUT/rc.txt
