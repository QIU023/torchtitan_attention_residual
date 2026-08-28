#!/bin/bash
# MoonEP's own tests at 2 ranks (written for 8). MOONEP_MEM_HANDLE_TYPE from env (auto|fd|fabric).
S=<scratch>; M=$S/moonep
source /venv/main/bin/activate; cd $M
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH NCCL_NVLS_ENABLE=0
export MOONEP_MEM_HANDLE_TYPE=${MOONEP_MEM_HANDLE_TYPE:-auto}
echo "handle_type=$MOONEP_MEM_HANDLE_TYPE"
for t in test_planning test_dispatch test_combine test_e2e test_grad_reduce test_prefetch; do
  echo "### $t"
  timeout 900 torchrun --nproc_per_node=2 --master_port=41400 -m pytest -x -q tests/$t.py 2>&1 | grep -v "^\s*$" | grep -E "passed|failed|error|Error|skipped|assert" | grep -v "^\[rank1\]" | tail -4 | cut -c1-220
done
echo "### SELFTESTS_DONE"
