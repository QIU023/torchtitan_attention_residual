#!/bin/bash
# after the H100 chain: the tower benchmark's time crossover points (448 px, 1456 px) under the recipe's SAC
until grep -q CHAIN_DONE /workspace/chain.txt 2>/dev/null; do sleep 30; done
ACS=sac CASES_LIST="448px 1456px" /workspace/kit_cpmm/run_tower_bench.sh
echo "$(date +%H:%M:%S) xover done" >> /workspace/chain.txt
