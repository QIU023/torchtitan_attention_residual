#!/bin/bash
# After probe_params.sh (queued by after_moonep_params.sh), run moonep_rows_check.sh; one GPU job at a time.
until grep -q "probe_params rc=" ~/mep/results/chain.txt 2>/dev/null; do sleep 30; done
echo "$(date +%H:%M:%S) moonep rows check start" >> ~/mep/results/chain.txt
bash ~/kit/kit_h100_2026-09-30/moonep_rows_check.sh > ~/mep/results/moonep_rows.out 2>&1
echo "$(date +%H:%M:%S) moonep rows check rc=$?" >> ~/mep/results/chain.txt
