#!/bin/bash
# The user (09-29 07:55): "先做PR A再MoonEP". PR A (4312's matrix first), then MoonEP, then DEP, on the
# source-built torch.
M=~/mep; Q=$M/results/queue.txt
note() { echo "$(date +%H:%M:%S) $*" >> $Q; }
note "queue2: PR A, MoonEP, DEP"
VENV=$M/venv_src bash ~/kit/pra/pra.sh > $M/pra.log 2>&1; note "pra rc=$?"
rm -rf $M/results/moonep/ci $M/results/moonep/cache*
VENV=$M/venv_src bash ~/kit/moonep/smoke_h100.sh > $M/moonep_smoke.log 2>&1; note "moonep smoke rc=$?"
VENV=$M/venv_src bash ~/kit/dep/dep4.sh > $M/dep4.log 2>&1; note "dep rc=$?"
note "queue done"
