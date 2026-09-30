#!/bin/bash
# After moonep_rows_check.sh (queued by after_params_rows.sh), run probe_stages.sh; one GPU job at a time.
until grep -q "moonep rows check rc=" ~/mep/results/chain.txt 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) probe_stages start" >> ~/mep/results/chain.txt
bash ~/kit/kit_h100_2026-09-30/probe_stages.sh > ~/mep/results/probe_stages.out 2>&1
echo "$(date +%H:%M:%S) probe_stages rc=$?" >> ~/mep/results/chain.txt
