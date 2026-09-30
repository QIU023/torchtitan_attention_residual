#!/bin/bash
# After the DEP campaign on the 09-30 box, one GPU job at a time: rerun the r10 M16 traces on a quiet box, the text-only
# step time diagnostics (diag_k0.sh), the DEP tree's CPU and GPU tests, then the MoonEP smoke (tests, CI cell, numerics, timing) and the MoonEP load cells.
# Progress in ~/mep/results/chain.txt.
until [ -f ~/mep/results/dep_h100_pp4vpp4/DONE ]; do sleep 60; done
C=~/mep/results/chain.txt
echo "$(date +%H:%M:%S) r10 M16 trace rerun start" >> $C
LEVEL="r10:2048:1:3.61:16:7" bash ~/kit/kit_h100_2026-09-30/rerun_traces.sh > ~/mep/results/rerun_traces.out 2>&1
echo "$(date +%H:%M:%S) rerun rc=$?; text-only diagnostics start" >> $C
bash ~/kit/kit_h100_2026-09-30/diag_k0.sh > ~/mep/results/diag_k0.out 2>&1
echo "$(date +%H:%M:%S) diag rc=$?; DEP tests start" >> $C
mkdir -p ~/mep/results/dep_tests
( cd ~/mep/w/dep && timeout 1200 ~/mep/venv_src/bin/python -m pytest tests/unit_tests/cpu/test_kimi_k3_dep_plan.py \
    tests/unit_tests/cpu/test_kimi_k3_vision_dep.py -q > ~/mep/results/dep_tests/cpu.log 2>&1 )
echo "$(date +%H:%M:%S) DEP cpu rc=$? $(tail -1 ~/mep/results/dep_tests/cpu.log)" >> $C
( cd ~/mep/w/dep && timeout 1200 ~/mep/venv_src/bin/python -m pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q \
    > ~/mep/results/dep_tests/gpu.log 2>&1 )
echo "$(date +%H:%M:%S) DEP gpu rc=$? $(tail -1 ~/mep/results/dep_tests/gpu.log); moonep smoke start" >> $C
VENV=~/mep/venv_src TREE=~/mep/w/moonep MOONEP=~/mep/MoonEP_src OUT=~/mep/results/moonep_smoke \
  bash ~/kit/kit_moonep_rewrite_2026-09-29/smoke.sh > ~/mep/results/moonep_smoke.out 2>&1
echo "$(date +%H:%M:%S) moonep smoke rc=$?; load start" >> $C
MODE=real TREE=~/mep/w/moonep VENV=~/mep/venv_src OUT=~/mep/results/moonep_load \
  bash ~/kit/overnight/moonep/run_load.sh > ~/mep/results/moonep_load.out 2>&1
echo "$(date +%H:%M:%S) moonep load rc=$?" >> $C
