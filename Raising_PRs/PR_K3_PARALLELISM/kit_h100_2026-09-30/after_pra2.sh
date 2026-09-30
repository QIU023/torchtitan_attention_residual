#!/bin/bash
# PR A rerun with expandable segments (the first run OOMed on fragmentation); the tests already ran (70 passed on both).
C=~/mep/results/chain.txt
echo "$(date +%H:%M:%S) pra campaign (expandable segments) start" >> $C
bash ~/kit/kit_h100_2026-09-30/pra/run_pra_h100.sh pp4vp2 pp4vp4 dp2pp2vp2 pp2vp2 pp2vp4 > ~/mep/results/pra_h100.out 2>&1
echo "$(date +%H:%M:%S) pra campaign rc=$?" >> $C
