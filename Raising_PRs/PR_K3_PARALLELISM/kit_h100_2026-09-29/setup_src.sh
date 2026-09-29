#!/bin/bash
# The source-built torch's venv: torch from ~/mep/pt/dist, the Triton its commit pins, torchvision built
# against it, main's requirements, the cutlass DSL family at 4.6.2 on the CUDA 12 bindings (the 13.x
# bindings need a 580 driver), mooncake for #4764's remote backend, and MoonEP 33327eb in its own checkout.
set -x
M=~/mep; . $M/venv_src/bin/activate; cd $M/pt
uv pip install dist/torch-*.whl
TP=$(cat .ci/docker/ci_commit_pins/triton.txt); TV=$(cat .ci/docker/triton_version.txt)
uv pip install --pre "triton==$TV+git${TP:0:8}" --index-url https://download.pytorch.org/whl/nightly/ \
  || uv pip install --pre "pytorch-triton==$TV+git${TP:0:8}" --index-url https://download.pytorch.org/whl/nightly/
python -c "import torch, triton; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), 'triton', triton.__version__)"
python -c "import torch.distributed.config as c; print('per_edge', hasattr(c, 'pipeline_per_edge_p2p'))"
cd $M; [ -d vision ] || git clone -q --depth 1 -b nightly https://github.com/pytorch/vision.git
cd vision; git log -1 --format="%H %s" > $M/vision_commit.txt
export CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0
FORCE_CUDA=1 MAX_JOBS=64 uv pip install --no-build-isolation -v . > $M/vision_build.log 2>&1; echo "vision rc=$?"
cd $M
uv pip install -r w/main_5dc9/.ci/docker/requirements.txt -r w/main_5dc9/.ci/docker/requirements-vlm.txt
uv pip install pytest pytest-subtests expecttest pyflakes mooncake-transfer-engine==0.3.13.post1
V=4.6.2
uv pip install nvidia-cutlass-dsl==$V nvidia-cutlass-dsl-libs-base==$V nvidia-cutlass-dsl-libs-core==$V \
  nvidia-cutlass-dsl-libs-cu12==$V "cuda-bindings>=12.9.4,<13" "cuda-python>=12.9,<13"
[ -d MoonEP_src ] || git clone -q https://github.com/MoonshotAI/MoonEP.git MoonEP_src
git -C MoonEP_src checkout -q 33327eb
(cd MoonEP_src && TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 uv pip install -e . --no-build-isolation) > $M/moonep_build_src.log 2>&1
echo "moonep rc=$?"
python -c "import torch; print('torch', torch.__version__)"
python -c "import torchvision; print('torchvision', torchvision.__version__)"
python -c "import moonep; print('moonep', moonep.__file__)"
python -c "import cuda.bindings.runtime as rt; print('cudart devices', rt.cudaGetDeviceCount())"
python -c "import cutlass; print('cutlass ok')"
uv pip list 2>/dev/null | grep -iE "^torch|^triton|cutlass|attn|moonep|mooncake|spmd|remat|cuda-bindings|cuda-python"
echo SETUP_SRC_DONE
