#!/bin/bash
# #4380 checks on the PR head 6317c5538 (H100, 4 GPUs): the GPU unit test, the CPU planner test, and pyrefly on the PR
# head against main 948d65c86 (the PR's files only count). torch 0906 needs the kit's import shims (mpp, pipelining names).
O=/workspace/h100_unit; mkdir -p $O; L=$O/summary.txt
. /workspace/kit_setup/env_cu126.sh
cd /workspace/tt_cpmm
echo "head $(git rev-parse HEAD) $(git status --short | wc -l) dirty" > $L
( PYTHONPATH=/workspace/kit_setup/shims:. CUDA_VISIBLE_DEVICES=0,1,2,3 DISTRIBUTED_TESTS_DEFAULT_TIMEOUT=3000 \
  TORCHINDUCTOR_CACHE_DIR=$O/ic TRITON_CACHE_DIR=$O/tc timeout 3600 python -m pytest -v -p no:cacheprovider \
  tests/unit_tests/gpu/test_kimi_k3_vision_cp.py > $O/gpu_test.log 2>&1; echo "gpu unit rc=$? $(tail -1 $O/gpu_test.log)" >> $L ) &
( PYTHONPATH=/workspace/kit_setup/shims:. CUDA_VISIBLE_DEVICES= timeout 900 python -m pytest -q -p no:cacheprovider \
  tests/unit_tests/cpu/test_kimi_k3_vision_cp_plan.py > $O/cpu_plan.log 2>&1; echo "cpu plan rc=$? $(tail -1 $O/cpu_plan.log)" >> $L ) &
( for t in tt_cpmm tt_main; do cd /workspace/$t && timeout 1800 pyrefly check --python-interpreter $(which python) > $O/pyrefly_$t.log 2>&1; echo "pyrefly $t rc=$? $(grep -c '^ERROR' $O/pyrefly_$t.log) errors" >> $L; done ) &
wait
echo UNIT_DONE >> $L
