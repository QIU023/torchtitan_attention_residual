#!/bin/bash
# After the #4656 matrix and the CPU tests: MoonEP, DEP and PR A (with the #4765 / #4764 smokes) in turn,
# all on the source-built torch.
M=~/mep; Q=$M/results/queue.txt
note() { echo "$(date +%H:%M:%S) $*" >> $Q; }
until [ -f $M/results/m4656/DONE ]; do sleep 20; done; note "m4656 done"
until grep -q "^done" $M/results/cpu_tests/summary.txt 2>/dev/null; do sleep 20; done; note "cpu tests done"
VENV=$M/venv_src bash ~/kit/moonep/smoke_h100.sh > $M/moonep_smoke.log 2>&1; note "moonep smoke rc=$?"
VENV=$M/venv_src bash ~/kit/dep/dep4.sh > $M/dep4.log 2>&1; note "dep rc=$?"
VENV=$M/venv_src bash ~/kit/pra/pra.sh > $M/pra.log 2>&1; note "pra rc=$?"
note "queue done"
