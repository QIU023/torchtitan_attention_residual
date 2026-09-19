#!/bin/bash
# DeepEP v2 on this box. Every marker is gated on a real import, never echoed blind.
set -x
source /workspace/kit/h200/env.sh
L=/workspace/results/deepep_install.log
exec > >(tee -a $L) 2>&1
echo "=== $(date) install_deepep start ==="

OLD_NCCL=$(python -c "import torch;print('.'.join(map(str,torch.cuda.nccl.version())))")
echo "NCCL before: $OLD_NCCL"

uv pip install --python /workspace/venv_k3/bin/python "nvidia-nccl-cu12==2.31.2" || { echo "DEEPEP_STAGE_NCCL_FAILED"; exit 1; }

python - <<'PY' || { echo "DEEPEP_STAGE_NCCL_FAILED"; exit 1; }
import ctypes, os, torch
# torch.cuda.nccl.version() is the compile-time constant; ask the library that
# is actually loaded, which is the one DeepEP will link against.
lib = os.path.normpath(os.path.join(os.path.dirname(torch.__file__), "..", "nvidia", "nccl", "lib", "libnccl.so.2"))
v = ctypes.c_int()
ctypes.CDLL(lib).ncclGetVersion(ctypes.byref(v))
n = v.value
print(f"runtime NCCL {n//10000}.{(n%10000)//100}.{n%100} at {lib}")
assert n >= 23004, n
PY
echo "DEEPEP_STAGE_NCCL_OK"

# torch itself must still work on the new NCCL before anything is built on it.
torchrun --nproc_per_node=4 --master_port=44010 /workspace/kit/_nccl_smoke.py || { echo "DEEPEP_STAGE_TORCH_FAILED"; exit 1; }
echo "DEEPEP_STAGE_TORCH_OK"

cd /workspace/src/DeepEP
TORCH_CUDA_ARCH_LIST=9.0 python setup.py install 2>&1 | tail -30

python -c "
from deep_ep import ElasticBuffer
import deep_ep, os
print('deep_ep at', os.path.dirname(deep_ep.__file__))
print('ElasticBuffer', ElasticBuffer)
" || { echo "DEEPEP_STAGE_BUILD_FAILED"; exit 1; }
echo "DEEPEP_STAGE_BUILD_OK"
echo "=== $(date) install_deepep end ==="
