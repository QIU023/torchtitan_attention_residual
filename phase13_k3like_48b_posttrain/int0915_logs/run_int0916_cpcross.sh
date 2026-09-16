#!/bin/bash
# Dynamic CP on the 09-16 integration tree crossed with pp2 and tp2 (four GPUs each, seeded, 3 steps, tower cut at threshold 96).
set -uo pipefail
T=/tmp/wt_int0916; S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
OUT=/workspace/.smoke_int0916/cpcross_$(date +%H%M%S); mkdir -p $OUT
run(){ local nm=$1 gpus=$2 cfg=$3; local d=$OUT/$nm; mkdir -p $d
  ( source /workspace/venv_bfx9/bin/activate && cd $T && CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=$d/triton TORCHINDUCTOR_CACHE_DIR=$d/inductor \
    PYTHONPATH=$S:/workspace/pylib/attn_gym_main:$T timeout 2400 torchrun --nproc_per_node=4 --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module cpmm_probe --config $cfg --metrics.log_freq 1 --training.steps 3 --debug.seed 42 --debug.deterministic --dump-folder $d > $d/log.txt 2>&1 )
  echo "$nm rc=$?" >> $OUT/rc.txt
  sed 's/\x1b\[[0-9;]*m//g' $d/log.txt | grep -oE 'step: *[0-9]+ +loss: *[0-9.]+ +grad_norm: *[0-9.]+' | sort -u | tr -s ' ' | paste -sd'|' >> $OUT/rc.txt
  sed 's/\x1b\[[0-9;]*m//g' $d/log.txt | grep -E "Error|Dynamic CP" | grep -v lspci | awk '!s[$0]++' | head -3 | cut -c1-200 >> $OUT/rc.txt
  rm -rf $d/triton $d/inductor; }
run allgather_cp2_pp2 0,1,2,3 kimi_k3_mm_allgather_kv_cp2_pp2_min96 &
run allgather_cp2_tp2 4,5,6,7 kimi_k3_mm_allgather_kv_cp2_tp2_min96 &
wait
run ulysses_cp2_pp2 0,1,2,3 kimi_k3_mm_ulysses_cp2_pp2_min96 &
run ulysses_cp2_tp2 4,5,6,7 kimi_k3_mm_ulysses_cp2_tp2_min96 &
wait; echo DONE >> $OUT/rc.txt
