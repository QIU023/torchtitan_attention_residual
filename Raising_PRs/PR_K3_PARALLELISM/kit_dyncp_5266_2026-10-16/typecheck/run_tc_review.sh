#!/bin/bash
# The two typecheck CP cells on the final fix commit with the split probe, GPUs 0,1 and 6,7, sequential runner mode.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; T=$S/wt_cpmm_fix; D=$S/dyncp_fix/tc3
. /workspace/venv_1006/bin/activate
echo "$(date +%H:%M:%S) tree $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)" >> $D/progress.txt
one() {  # <test_name> <gpus>
  ( cd $T && SPLIT_PROBE=1 CUDA_VISIBLE_DEVICES=$2 PYTHONPATH=$S/dyncp_fix/split_probe:$S/main_pins_1016:$T TORCHINDUCTOR_CACHE_DIR=$S/dyncp_fix/tc/cache_$1/ic TRITON_CACHE_DIR=$S/dyncp_fix/tc/cache_$1/tc \
    timeout 3000 python -m tests.integration_tests.run_tests $D/out --test_suite h100 --test_name $1 --ngpu 2 --no-parallel > $D/$1.log 2>&1 )
  local clean=$(sed 's/\x1b\[[0-9;]*m//g' $D/$1.log)
  echo "$(date +%H:%M:%S) $1 failed=$(echo "$clean" | grep -c 'torchtitan.train FAILED') last step=$(echo "$clean" | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1) typeerr=$(echo "$clean" | grep -c SpmdTypeError) installed=$(echo "$clean" | grep -c 'SPLIT_PROBE installed') splits=$(echo "$clean" | grep -c 'SPLIT_PROBE rank')" >> $D/progress.txt
}
one kimi_k3_mm_allgather_kv_cp 0,1 &
one kimi_k3_mm_ulysses_cp 6,7 &
wait
echo "$(date +%H:%M:%S) done" >> $D/progress.txt
