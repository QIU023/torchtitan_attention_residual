#!/bin/bash
# The ported memory32 script, debug-model copy (standard dispatcher, EP8), on the release-format fixture.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; T=$S/wt_mxqat_fix; V=$S/verify_stack_1013
until [ -f $V/elfie_entry_run8_loss/DONE ]; do sleep 20; done
E=$V/memory32_smoke; FIX=$V/elfie_entry/fixture
. /workspace/venv_1006/bin/activate
export PYTHONPATH=$S/main_pins:$T:$T/scripts/validation/kimi_k3:$S/elfie_ao
export NCCL_RUNTIME_CONNECT=0 NCCL_NET_PLUGIN=none PYTORCH_ALLOC_CONF=expandable_segments:True TORCH_NCCL_ASYNC_ERROR_HANDLING=1 OMP_NUM_THREADS=8
cd $T
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 timeout 2400 torchrun --nproc-per-node=8 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
  $E/hybridep_stage_memory_debug.py --checkpoint $FIX --output $E/out > $E/run.log 2>&1
echo "$(date -u +%H:%M:%S) rc=$? pass=$(grep -c HYBRIDEP_STAGE_MEMORY_PASS $E/run.log) errors=$(grep -cE '(Error|Exception):' $E/run.log)" > $E/progress.txt
echo done > $E/DONE
