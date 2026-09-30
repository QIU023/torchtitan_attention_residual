#!/bin/bash
# After after_dep.sh ends (its last chain.txt line is the MoonEP load rc), run probe_step1.sh; one GPU job at a time.
until grep -q "moonep load rc=" ~/mep/results/chain.txt 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) probe_step1 start" >> ~/mep/results/chain.txt
bash ~/kit/kit_h100_2026-09-30/probe_step1.sh > ~/mep/results/probe_step1.out 2>&1
echo "$(date +%H:%M:%S) probe_step1 rc=$?" >> ~/mep/results/chain.txt
