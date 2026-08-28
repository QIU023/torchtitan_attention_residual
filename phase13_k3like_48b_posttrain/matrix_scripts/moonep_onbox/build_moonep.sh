#!/bin/bash
# MoonEP (MoonshotAI/MoonEP master @ 2bd860b) into /venv/main, against torch nightly + CUDA 13.0.
# Run only after the EP matrix has finished: it adds nvidia-cutlass-dsl to the venv.
set -ux
source /venv/main/bin/activate
M=<scratch>/moonep
cd $M && git log --oneline -1
export CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=48
uv pip install "nvidia-cutlass-dsl==4.4.2" 2>&1 | tail -3
time python -m pip install --no-build-isolation --no-deps -e . 2>&1 | tail -5
cd / && python -c "import moonep, torch; from moonep import Buffer, MoonEPCommPlan; from moonep.buffer import create_nvl_single_owner_tensor; print('MOONEP_OK', moonep.__file__, '| torch', torch.__version__)"
echo "MOONEP_BUILD_EXIT=$?"
