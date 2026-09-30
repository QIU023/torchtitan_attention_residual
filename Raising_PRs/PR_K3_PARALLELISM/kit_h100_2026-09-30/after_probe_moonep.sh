#!/bin/bash
# After probe_step1 (queued by after_chain_probe.sh), run moonep_rerun.sh; one GPU job at a time.
until grep -q "probe_step1 rc=" ~/mep/results/chain.txt 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) moonep rerun start" >> ~/mep/results/chain.txt
bash ~/kit/kit_h100_2026-09-30/moonep_rerun.sh > ~/mep/results/moonep_rerun.out 2>&1
echo "$(date +%H:%M:%S) moonep rerun rc=$?" >> ~/mep/results/chain.txt
