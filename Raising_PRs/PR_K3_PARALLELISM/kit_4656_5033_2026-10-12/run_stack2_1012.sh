#!/bin/bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; A=$S/m4656_5033/stack2; T=$S/wt_mxqat
. /workspace/venv_1006/bin/activate
echo "$(date +%H:%M:%S) head $(git -C $T rev-parse --short HEAD) dirty=$(git -C $T status --short | wc -l)" >> $A/progress.txt
( cd $T && PYTHONPATH=$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$A/cache/ic TRITON_CACHE_DIR=$A/cache/tc CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 LOG_RANK=0,1,2,3,4,5,6,7 \
  MODULE=tests.integration_tests.b200 CONFIG=kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4 timeout 3000 ./run_train.sh --output-dir $A/out > $A/dep_cell.log 2>&1 )
echo "$(date +%H:%M:%S) dep cell rc=$? last step=$(sed 's/\x1b\[[0-9;]*m//g' $A/dep_cell.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)" >> $A/progress.txt; rm -rf $A/out
( cd $T && PYTHONPATH=$T:$S/elfie_ao TORCHINDUCTOR_CACHE_DIR=$A/cache/ic TRITON_CACHE_DIR=$A/cache/tc CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 3000 \
  python -m pytest -q -p no:cacheprovider tests/unit_tests/gpu/models/kimi_k3/test_kimi_k3_attention_residual.py tests/unit_tests/gpu/recipes/test_kimi_attention_residual_override.py \
  tests/unit_tests/gpu/test_kimi_k3_vision_dep.py tests/unit_tests/gpu/test_kimi_k3_vision_cp.py tests/unit_tests/gpu/test_moonep.py > $A/gpu_tests.log 2>&1 )
echo "$(date +%H:%M:%S) gpu tests rc=$? $(tail -1 $A/gpu_tests.log)" >> $A/progress.txt
echo done > $A/DONE
