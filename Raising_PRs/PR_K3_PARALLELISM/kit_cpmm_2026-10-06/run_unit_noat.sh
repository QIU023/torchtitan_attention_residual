#!/bin/bash
# the GPU unit test with flex compiled without max-autotune inside the test (the proposed test commit), cold caches as in CI
O=/workspace/h100_unit2; rm -rf $O; mkdir -p $O; L=$O/summary.txt
. /workspace/kit_setup/env_cu126.sh
rm -rf /workspace/tt_cpmm_t && cp -r /workspace/tt_cpmm /workspace/tt_cpmm_t
cp /workspace/kit_setup/test_kimi_k3_vision_cp_noat.py /workspace/tt_cpmm_t/tests/unit_tests/gpu/test_kimi_k3_vision_cp.py
cd /workspace/tt_cpmm_t && echo "base $(git rev-parse HEAD) diff: $(git diff --stat | tail -1)" > $L
start=$(date +%s)
PYTHONPATH=/workspace/kit_setup/shims:. CUDA_VISIBLE_DEVICES=0,1,2,3 DISTRIBUTED_TESTS_DEFAULT_TIMEOUT=3000 \
  TORCHINDUCTOR_CACHE_DIR=$O/ic TRITON_CACHE_DIR=$O/tc timeout 3600 python -m pytest -v -p no:cacheprovider --durations=0 \
  tests/unit_tests/gpu/test_kimi_k3_vision_cp.py > $O/gpu_test.log 2>&1
echo "gpu unit rc=$? wall $(( $(date +%s) - start )) s: $(tail -1 $O/gpu_test.log)" >> $L
echo UNIT2_DONE >> $L
