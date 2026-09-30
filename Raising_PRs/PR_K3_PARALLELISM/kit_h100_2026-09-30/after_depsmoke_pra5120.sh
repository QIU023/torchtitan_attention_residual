#!/bin/bash
# After the DEP smoke recheck: PR A's three layouts that OOM at dim 6144 (two pipeline ranks each), at the sizing doc's
# fallback width 5120; results in ~/mep/results/pra_h100_d5120.
until grep -q "dep smoke (expandable segments) done" ~/mep/results/chain.txt 2>/dev/null; do sleep 60; done
C=~/mep/results/chain.txt; echo "$(date +%H:%M:%S) pra dim 5120 start" >> $C
OUT=~/mep/results/pra_h100_d5120 PPMEM_DIM=5120 bash ~/kit/kit_h100_2026-09-30/pra/run_pra_h100.sh dp2pp2vp2 pp2vp2 pp2vp4 > ~/mep/results/pra_h100_d5120.out 2>&1
echo "$(date +%H:%M:%S) pra dim 5120 rc=$?" >> $C
