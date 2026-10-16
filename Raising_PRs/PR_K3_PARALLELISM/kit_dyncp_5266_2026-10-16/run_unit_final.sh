#!/bin/bash
# GPU unit tests on the final trees before the stack matrix: review f4010b98c on GPUs 0-3, stack on GPUs 4-7 (warm caches).
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
. /workspace/venv_1006/bin/activate
one() {  # <name> <tree> <gpus> <pins> <cache> <tests...>
  local name=$1 tree=$2 gpus=$3 pins=$4 cache=$5; shift 5; local D=$S/dyncp_fix/$name; mkdir -p $D
  echo "tree $(git -C $tree rev-parse --short HEAD) dirty=$(git -C $tree status --short | wc -l)" > $D/tree.txt
  ( cd $tree && CUDA_VISIBLE_DEVICES=$gpus PYTHONPATH=$pins:$tree TORCHINDUCTOR_CACHE_DIR=$cache/ind TRITON_CACHE_DIR=$cache/tri \
    timeout 2400 python -m pytest -q -p no:cacheprovider -rA "$@" > $D/pytest.log 2>&1; echo "rc=$?" > $D/result.txt )
}
one gpu_fix_final $S/wt_cpmm_fix 0,1,2,3 $S/main_pins_1016 $S/dyncp_fix/gpu_fix tests/unit_tests/gpu/test_kimi_k3_vision_cp.py &
one gpu_stack_final $S/wt_mxqat_fix 4,5,6,7 $S/main_pins $S/dyncp_fix/gpu_stack tests/unit_tests/gpu/test_kimi_k3_vision_cp.py tests/unit_tests/gpu/test_kimi_k3_vision_dep.py &
wait
