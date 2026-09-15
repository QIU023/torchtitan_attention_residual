#!/bin/bash
# Upstream PR 4639 head as-is (KDA guard lifted locally): do its K3 CP recipes pass on this box?
set -uo pipefail
T=/tmp/wt_4639; OUT=/workspace/.smoke_int0915/upstream4639_$(date +%H%M%S); mkdir -p $OUT
run(){ local nm=$1 gpus=$2 cfg=$3; local d=$OUT/$nm; mkdir -p $d
  ( source /workspace/venv_bfx9/bin/activate && cd $T && CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=$d/triton TORCHINDUCTOR_CACHE_DIR=$d/inductor \
    PYTHONPATH=/workspace/pylib/attn_gym_main:$T timeout 1800 torchrun --nproc_per_node=2 --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module torchtitan_recipes.tests.b200 --config $cfg --metrics.log_freq 1 --training.steps 3 --dump-folder $d > $d/log.txt 2>&1 )
  echo "$nm rc=$?" >> $OUT/rc.txt
  sed 's/\x1b\[[0-9;]*m//g' $d/log.txt | grep -E "step: *[123] |Error" | grep -v lspci | head -5 | cut -c1-220 >> $OUT/rc.txt
  rm -rf $d/triton $d/inductor; }
run allgather 0,1 kimi_k3_debugmodel_mm_allgather_kv_cp2 &
run ulysses 2,3 kimi_k3_debugmodel_mm_ulysses_cp2 &
wait; echo DONE >> $OUT/rc.txt
