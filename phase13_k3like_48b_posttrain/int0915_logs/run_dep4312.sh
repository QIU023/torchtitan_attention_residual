#!/bin/bash
# DEP rebased onto PR 4312 (KDA guard lifted locally): pp8 x vp4 with and without the tower stage, 8 GPUs each, 3 steps.
set -uo pipefail
T=/tmp/wt_dep_4312; OUT=/workspace/.smoke_int0915/dep4312_$(date +%H%M%S); mkdir -p $OUT
run(){ local nm=$1 cfg=$2; local d=$OUT/$nm; mkdir -p $d
  ( source /workspace/venv_bfx9/bin/activate && cd $T && CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 TRITON_CACHE_DIR=$d/triton TORCHINDUCTOR_CACHE_DIR=$d/inductor \
    PYTHONPATH=/workspace/pylib/attn_gym_main:$T timeout 2400 torchrun --nproc_per_node=8 --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module torchtitan_recipes.tests.b200 --config $cfg --metrics.log_freq 1 --training.steps 3 --dump-folder $d > $d/log.txt 2>&1 )
  echo "$nm rc=$?" >> $OUT/rc.txt
  sed 's/\x1b\[[0-9;]*m//g' $d/log.txt | grep -E "step: *[123] |Error|vit_dep|tower stage" | grep -v lspci | awk '!s[$0]++' | head -6 | cut -c1-200 >> $OUT/rc.txt
  rm -rf $d/triton $d/inductor; }
run vit_dep kimi_k3_debugmodel_pp8_vp4_vit_dep
run pp8vp4 kimi_k3_debugmodel_pp8_vp4
echo DONE >> $OUT/rc.txt
