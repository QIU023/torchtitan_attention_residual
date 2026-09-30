#!/bin/bash
# Recheck MoonEP's per-rank rows with the fixed probe (zero_fill_ranges column 1 is a row count; nonzero dispatched rows):
# the load recipe's MoonEP cell, natural and skewed routing, AC off, 5 steps each. Results in ~/mep/results/moonep_rows.
M=~/mep; V=$M/venv_src; T=$M/w/moonep; KL=~/kit/overnight/moonep; O=$M/results/moonep_rows; rm -rf $O; mkdir -p $O
export MOONEP_AC=none
C=$O/cache; mkdir -p $C
for x in nat:0 skew:0.05; do
  tag=${x%%:*}; skew=${x#*:}; nobias=0; [ "$skew" != 0 ] && nobias=1; D=$O/num_moonep_$tag; mkdir -p $D
  ( cd $T && PYTHONPATH=$KL:. TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc LOAD_OUT=$D/load LOAD_SKEW=$skew LOAD_NO_BIAS=$nobias \
    timeout 1800 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module moonep_load --config load_moonep --training.steps 5 --metrics.log-freq 1 \
    --debug.seed 42 --debug.deterministic --dump-folder $D/out > $D/run.log 2>&1 )
  echo "$tag rc=$?" >> $O/summary.txt; rm -rf $D/out
done
$V/bin/python $KL/tab_load.py $O >> $O/summary.txt 2>&1
rm -rf $C; echo CHECK_DONE >> $O/summary.txt
