#!/bin/bash
# Usage: NP=2 run_hot_probe.sh
source /workspace/kit/h200/env.sh; NP=${NP:-2}; cd $TITAN
timeout 900 torchrun --nproc_per_node=$NP --master_port=$((41600+NP)) /workspace/kit/h200/moonep_hot_probe.py > /workspace/results/hot_probe_np$NP.log 2>&1; rc=$?
grep -a "HOT_PROBE" /workspace/results/hot_probe_np$NP.log | sed 's/\x1b\[[0-9;]*m//g'; echo "rc=$rc"; grep -a -m2 "Error\|error:" /workspace/results/hot_probe_np$NP.log | cut -c1-200
