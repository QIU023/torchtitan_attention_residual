#!/bin/bash
# DEP with triton 3.8 in venv_src (triton 3.9 kernels do not load on driver 560), then the #4380 tower table's PR column
# (the first pass left the PR's threshold unset) once the fixed bench is synced (marker TOWER_KIT_OK); never both at once.
C=/workspace/chain.txt
note() { echo "$(date +%H:%M:%S) $*" >> $C; }
note "chain3: DEP starts (triton $(/workspace/venv_src/bin/python -c 'import triton; print(triton.__version__)'))"
/workspace/kit_dep/dep_tables.sh
note "DEP finished: $(tail -1 /workspace/h100_dep/progress.txt)"
until [ -f /workspace/TOWER_KIT_OK ]; do sleep 30; done
note "tower PR column starts"
O=/workspace/h100_tower ONLY=pr ACS=sac CASES_LIST="2016px 4032px 4x2016px video16f448" /workspace/kit_cpmm/run_tower_bench.sh
O=/workspace/h100_tower ONLY=pr ACS="none full" CASES_LIST="4032px" /workspace/kit_cpmm/run_tower_bench.sh
note "CHAIN3_DONE"
