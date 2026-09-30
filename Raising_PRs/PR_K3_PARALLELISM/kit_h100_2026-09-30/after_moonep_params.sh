#!/bin/bash
# After moonep_rerun.sh (queued by after_probe_moonep.sh), run probe_params.sh; one GPU job at a time.
until grep -q "moonep rerun rc=" ~/mep/results/chain.txt 2>/dev/null; do sleep 60; done
echo "$(date +%H:%M:%S) probe_params start" >> ~/mep/results/chain.txt
bash ~/kit/kit_h100_2026-09-30/probe_params.sh > ~/mep/results/probe_params.out 2>&1
echo "$(date +%H:%M:%S) probe_params rc=$?" >> ~/mep/results/chain.txt
