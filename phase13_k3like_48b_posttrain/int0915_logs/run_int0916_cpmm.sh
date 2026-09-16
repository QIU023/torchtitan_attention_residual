#!/bin/bash
# Dynamic CP on the 09-16 integration tree, seeded: the 192-patch debug image cut over the pair (threshold 96) against the tower replicated.
# the two mm CP recipes with the large image partitioned (default) and with the tower forced replicated.
set -uo pipefail
T=/tmp/wt_int0916; S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
OUT=/workspace/.smoke_int0915/int0916_cpmm_$(date +%H%M%S); mkdir -p $OUT
run(){ local nm=$1 gpus=$2 mod=$3 cfg=$4; local d=$OUT/$nm; mkdir -p $d
  ( source /workspace/venv_bfx9/bin/activate && cd $T && CUDA_VISIBLE_DEVICES=$gpus TRITON_CACHE_DIR=$d/triton TORCHINDUCTOR_CACHE_DIR=$d/inductor \
    PYTHONPATH=$S:/workspace/pylib/attn_gym_main:$T timeout 1800 torchrun --nproc_per_node=2 --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module $mod --config $cfg --metrics.log_freq 1 --training.steps 3 --debug.seed 42 --debug.deterministic --dump-folder $d > $d/log.txt 2>&1 )
  echo "$nm rc=$?" >> $OUT/rc.txt
  sed 's/\x1b\[[0-9;]*m//g' $d/log.txt | grep -E "step: *[123] |Error|Dynamic CP" | grep -v lspci | awk '!s[$0]++' | head -6 | cut -c1-200 >> $OUT/rc.txt
  rm -rf $d/triton $d/inductor; }
run allgather_part 0,1 cpmm_probe kimi_k3_mm_allgather_kv_cp2_min96 &
run ulysses_part 2,3 cpmm_probe kimi_k3_mm_ulysses_cp2_min96 &
run allgather_rep 4,5 cpmm_probe kimi_k3_mm_allgather_kv_cp2_replicated &
run ulysses_rep 6,7 cpmm_probe kimi_k3_mm_ulysses_cp2_replicated &
wait; echo DONE >> $OUT/rc.txt
