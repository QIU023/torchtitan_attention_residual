#!/bin/bash
# PR A on the 09-30 box, one GPU job at a time: the test plan's five files on both trees, then the dim 6144 memory campaign.
C=~/mep/results/chain.txt
echo "$(date +%H:%M:%S) pra tests start" >> $C
bash ~/kit/kit_h100_2026-09-30/pra/run_pra_tests.sh > ~/mep/results/pra_tests.out 2>&1
echo "$(date +%H:%M:%S) pra tests done; pra campaign start" >> $C
bash ~/kit/kit_h100_2026-09-30/pra/run_pra_h100.sh pp4vp2 pp4vp4 dp2pp2vp2 pp2vp2 pp2vp4 > ~/mep/results/pra_h100.out 2>&1
echo "$(date +%H:%M:%S) pra campaign rc=$?" >> $C
