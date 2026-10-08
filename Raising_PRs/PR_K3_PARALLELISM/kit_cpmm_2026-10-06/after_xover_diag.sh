#!/bin/bash
# after the crossover cells: the fp32 large split cases again, with flex compiled without max-autotune (kit_setup/shims_noat);
# the production compile autotunes every new fp32 shape and timed out at 4032 px (10-08)
until grep -q "xover done" /workspace/chain.txt 2>/dev/null; do sleep 30; done
. /workspace/kit_setup/env_cu126.sh
D=/workspace/h100_diag; K=/workspace/kit_cpmm
for cp in 4 2; do
  ( cd /workspace/tt_cpmm && PYTHONPATH=/workspace/kit_setup/shims_noat:. TORCHINDUCTOR_CACHE_DIR=$D/noat_ic TRITON_CACHE_DIR=$D/noat_tc \
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((cp - 1))) timeout 2400 torchrun --nproc_per_node=$cp --master_port=29713 $K/diag_big_cases.py > $D/noat_cp$cp.log 2>&1 )
  echo "$(date +%H:%M:%S) diag noat cp$cp rc=$? $(grep -c 'pass True' $D/noat_cp$cp.log) pass / $(grep -c BIG_CASE $D/noat_cp$cp.log)" >> /workspace/chain.txt
done
echo "$(date +%H:%M:%S) DIAG2_DONE" >> /workspace/chain.txt
