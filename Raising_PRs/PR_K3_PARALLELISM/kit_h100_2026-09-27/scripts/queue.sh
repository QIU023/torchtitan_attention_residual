#!/bin/bash
# The whole H100 queue, one group after another; each group writes G<n>_DONE.
K=~/k927/kit
bash $K/g1_dep.sh > ~/k927/g1.log 2>&1
bash $K/g2_attnres.sh > ~/k927/g2.log 2>&1
bash $K/g3_pra.sh > ~/k927/g3.log 2>&1
if ! grep -q "rc=0" ~/k927/res/pra/fit_b4656/train.log 2>/dev/null; then
  PPMEM_DIM=3072 bash $K/g3_pra.sh > ~/k927/g3_dim3072.log 2>&1
fi
bash $K/g4_balance.sh > ~/k927/g4.log 2>&1
bash $K/g5_moonep.sh > ~/k927/g5.log 2>&1
echo done > ~/k927/QUEUE_DONE
