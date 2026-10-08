#!/bin/bash
# chain on the H100 after the tower benchmark's sac/none pass: the full-AC pass, the fp32 large split cases (CP4, CP2),
# the end-to-end runs, then the CI fix smoke. Each step logs to /workspace/chain.txt.
L=/workspace/chain.txt; K=/workspace/kit_cpmm
note() { echo "$(date +%H:%M:%S) $*" >> $L; }
until grep -q TOWER_DONE /workspace/h100_tower/progress.txt 2>/dev/null; do sleep 20; done
note "tower sac/none done"
ACS=full $K/run_tower_bench.sh; note "tower full done"
. /workspace/kit_setup/env_cu126.sh
mkdir -p /workspace/h100_diag
for cp in 4 2; do
  ( cd /workspace/tt_cpmm && PYTHONPATH=/workspace/kit_setup/shims:. TORCHINDUCTOR_CACHE_DIR=/workspace/h100_diag/ic TRITON_CACHE_DIR=/workspace/h100_diag/tc \
    CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((cp - 1))) timeout 1800 torchrun --nproc_per_node=$cp --master_port=29712 $K/diag_big_cases.py > /workspace/h100_diag/cp$cp.log 2>&1 )
  note "diag cp$cp rc=$? $(grep -c 'pass True' /workspace/h100_diag/cp$cp.log) pass / $(grep -c BIG_CASE /workspace/h100_diag/cp$cp.log)"
done
$K/run_e2e_h100.sh; note "e2e done"
/workspace/kit_setup/run_muonfix_h100.sh; note "muonfix done"
note CHAIN_DONE
