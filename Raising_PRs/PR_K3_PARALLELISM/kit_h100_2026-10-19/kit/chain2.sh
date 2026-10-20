#!/bin/bash
# After the #4380 training table and the torch build (finished either way, so nothing else loads the CPUs): the #4380
# tower table alone on the box; then DEP, only on a successful build.
C=/workspace/chain.txt
note() { echo "$(date +%H:%M:%S) $*" >> $C; }
until grep -q "TRAIN_TABLE_DONE\|no seed" /workspace/h100_train2/progress.txt 2>/dev/null; do sleep 30; done
until grep -q "^BUILD_OK\|^BUILD_FAILED" /workspace/build_torch.log 2>/dev/null; do sleep 30; done
note "train table done, build $(grep -o '^BUILD_[A-Z]*.*' /workspace/build_torch.log | head -1); tower table starts"
O=/workspace/h100_tower ACS=sac CASES_LIST="2016px 4032px 4x2016px video16f448" /workspace/kit_cpmm/run_tower_bench.sh
O=/workspace/h100_tower ACS="none full" CASES_LIST="4032px" /workspace/kit_cpmm/run_tower_bench.sh
note "tower table done"
grep -q "^BUILD_OK" /workspace/build_torch.log || { note "no source-built torch, DEP not run"; exit 1; }
note "DEP starts"
/workspace/kit_dep/dep_tables.sh
note "CHAIN_DONE"
