#!/bin/bash
# After the PR A campaign: the DEP memory smoke again with expandable segments (titan's run_train.sh default), for the
# configurations that OOMed without it at 13:0x on 09-30: seq 4096 cap 3 M16 (off and bubble) and seq 6144 cap 2 M16.
until grep -q "pra campaign rc=" ~/mep/results/chain.txt 2>/dev/null; do sleep 60; done
C=~/mep/results/chain.txt; echo "$(date +%H:%M:%S) dep smoke (expandable segments) start" >> $C
R=~/mep/results; rm -rf $R/dep_smoke_pp4vpp4_noexp; cp -r $R/dep_smoke_pp4vpp4 $R/dep_smoke_pp4vpp4_noexp; rm -rf $R/dep_smoke_pp4vpp4_noexp/cache
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DEPR_LAYOUT=pp4vpp4
SEQ=4096 CAP=3 RATIO=3.42 MBS=16 bash ~/kit/kit_h100_2026-09-30/smoke_mem.sh > $R/smoke_exp_s4096.out 2>&1
SEQ=6144 CAP=2 RATIO=1.92 MBS=16 bash ~/kit/kit_h100_2026-09-30/smoke_mem.sh > $R/smoke_exp_s6144.out 2>&1
echo "$(date +%H:%M:%S) dep smoke (expandable segments) done" >> $C
