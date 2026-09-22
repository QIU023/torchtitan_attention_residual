#!/bin/bash
# MoonEP at the pin the PR names, built for Hopper. uv isolates build environments and
# MoonEP setup.py imports torch, so the build runs with --no-build-isolation.
set -uxo pipefail
export PATH=$HOME/.local/bin:$PATH
source ~/venv_k3/bin/activate
cd ~/MoonEP
git log --oneline -n 1 | cut -c1-60
grep -c "make_rmem_tensor" moonep/grad_reduce.py
uv pip install setuptools wheel ninja 2>&1 | tail -n 1
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 uv pip install -e . --no-build-isolation 2>&1 | tail -n 8
python -c "import moonep; print('moonep ok', moonep.__file__)"
echo MOONEP_DONE
