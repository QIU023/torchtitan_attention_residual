#!/bin/bash
# Waits for the seq 1536 ratio probe, then runs run_dep_new.sh at three cost ratios measured from K2.5 traces on this
# box (dep_new_calib.sh, probe_ratio.sh): seq 2048 with 224 px (0.116), seq 2048 with 1008 px (0.220), and seq 1536 with
# 1008 px (the probe's value), one image per sample, M16. Numerics run at the first level.
M=~/mep; O=$M/results/dep_new_calib
until [ -f $O/tr_s1536_r1024/DONE ]; do sleep 15; done
r3=$(grep -a -o 'cost ratio (trace, kernel time): [0-9.]*' $O/tr_s1536_r1024/ratio.txt | grep -o '[0-9.]*$')
if [ -z "$r3" ]; then echo "$(date +%H:%M:%S) no measured ratio at seq 1536; campaign not started" >> $O/progress.txt; exit 1; fi
export LEVELS="c012:2048:1:0.116:16:224 c022:2048:1:0.220:16:1024 c030:1536:1:$r3:16:1024"
bash ~/kit/kit_h100_2026-10-02/run_dep_new.sh
