#!/bin/bash
# The GPU unit suite the way CI runs it, on the stack and on main: the "not multi_gpu" lane on one GPU (both trees at
# once, own caches and MASTER_PORT), then the multi_gpu lane on all eight GPUs, one tree at a time. Writes DONE_GPU.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; V=$S/verify_stack_1013
lane() {  # <tree> <out> <marker expression> <gpus> <port>
  ( . /workspace/venv_1006/bin/activate; cd $1 && MASTER_PORT=$5 CUDA_VISIBLE_DEVICES=$4 PYTHONPATH=$S/main_pins:$1:$S/elfie_ao \
    TORCHINDUCTOR_CACHE_DIR=$2.cache/ic TRITON_CACHE_DIR=$2.cache/tc timeout 5400 python -m pytest -q -p no:cacheprovider -rfE \
    --continue-on-collection-errors -m "$3" --strict-markers --durations=40 tests/unit_tests/gpu > $2 2>&1; echo "rc=$?" >> $2 )
}
lane $S/wt_mxqat_fix $V/lane1_stack.txt "not multi_gpu" 0 29611 & lane $S/wt_mainref $V/lane1_main.txt "not multi_gpu" 1 29612 & wait
lane $S/wt_mxqat_fix $V/lanem_stack.txt multi_gpu 0,1,2,3,4,5,6,7 29613
lane $S/wt_mainref $V/lanem_main.txt multi_gpu 0,1,2,3,4,5,6,7 29614
echo done > $V/DONE_GPU
