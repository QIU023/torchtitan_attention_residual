#!/bin/bash
# After PR A's dim 5120 cells: DEP at the levels chosen before the fragmentation OOMs (with titan's expandable segments they
# fit): 90 : 10 = seq 6144 with up to 2 images per sample, 84 : 16 = seq 4096 with up to 3, pp4 x vpp4, M8 and M16, cost
# ratios from dep_calib_s16/levels.txt; timing, traces and fill only (numerics ran at seq 2048). Results in
# ~/mep/results/dep_h100_pp4vpp4_origlevels, apart from the seq 2048 campaign.
until grep -q "pra dim 5120 rc=" ~/mep/results/chain.txt 2>/dev/null; do sleep 60; done
C=~/mep/results/chain.txt; echo "$(date +%H:%M:%S) dep original levels start" >> $C
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=~/mep/results/dep_h100_pp4vpp4_origlevels NUMERICS=0 DEPR_LAYOUT=pp4vpp4 \
  LEVELS="s6144c2:6144:2:1.92:8,16 s4096c3:4096:3:3.42:8,16" \
  bash ~/kit/kit_h100_2026-09-30/run_dep_h100.sh > ~/mep/results/dep_origlevels.out 2>&1
echo "$(date +%H:%M:%S) dep original levels rc=$?" >> $C
