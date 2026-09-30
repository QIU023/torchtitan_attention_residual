#!/bin/bash
# 09-30 box: the venv around the source-built torch (68e0ae4, CUDA 12.8, sm90), with the 09-29 fix_src.sh lessons
# folded in: torchvision built here without resolving torch, the pinned Triton, main's requirements, then the
# source torch and the Triton pin put back without dependency resolution and the cu13 runtime packages dropped,
# the cutlass DSL family at 4.6.2 on the CUDA 12 bindings (13.x needs a 580 driver), MoonEP 33327eb.
set -x
M=~/mep; . $M/venv_src/bin/activate; cd $M
CU=/usr/local/cuda-12.8
uv pip install --no-deps $M/pt/dist/torch-*.whl
uv pip install --no-deps --pre "triton==3.8.0+gitc01b6774" --index-url https://download.pytorch.org/whl/nightly/
[ -d vision ] || git clone -q --depth 1 -b nightly https://github.com/pytorch/vision.git vision
git -C vision log -1 --format="%H %s" > $M/vision_commit.txt
(cd vision && CUDA_HOME=$CU PATH=$CU/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 FORCE_CUDA=1 MAX_JOBS=64 \
  uv pip install --no-build-isolation --no-deps -v .) > $M/vision_build.log 2>&1
echo "vision rc=$?"
uv pip install -r $M/w/dep/.ci/docker/requirements.txt -r $M/w/dep/.ci/docker/requirements-vlm.txt
uv pip install pytest pytest-subtests expecttest pyflakes
V=4.6.2
uv pip install nvidia-cutlass-dsl==$V nvidia-cutlass-dsl-libs-base==$V nvidia-cutlass-dsl-libs-core==$V \
  nvidia-cutlass-dsl-libs-cu12==$V nvidia-cutlass-dsl-libs-cu13==$V "cuda-bindings>=12.9.4,<13" "cuda-python>=12.9,<13"
uv pip install --reinstall --no-deps $M/pt/dist/torch-*.whl
uv pip install --reinstall --no-deps --pre "triton==3.8.0+gitc01b6774" --index-url https://download.pytorch.org/whl/nightly/
uv pip install --reinstall --no-deps nvidia-cutlass-dsl-libs-cu13==$V
for p in cuda-toolkit nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft \
  nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink \
  nvidia-nvshmem-cu13 nvidia-nvtx; do uv pip uninstall -q $p 2>/dev/null; done
[ -d MoonEP_src ] || git clone -q https://github.com/MoonshotAI/MoonEP.git MoonEP_src
git -C MoonEP_src checkout -q 33327eb
(cd MoonEP_src && CUDA_HOME=$CU PATH=$CU/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 \
  uv pip install -e . --no-build-isolation --no-deps) > $M/moonep_build.log 2>&1
echo "moonep rc=$?"
python -c "import torch; print('torch', torch.__version__, torch.version.git_version[:9], torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.nccl.version())"
python -c "import torch.distributed.config as c; print('per_edge', hasattr(c, 'pipeline_per_edge_p2p'))"
python -c "import inspect, torch.distributed.pipelining.schedules as s; print('unshard_lookahead', 'unshard_lookahead' in inspect.signature(s.ScheduleInterleaved1F1B.__init__).parameters)"
python -c "import triton; print('triton', triton.__version__)"
python -c "import torchvision, torch; print('torchvision', torchvision.__version__, torch.ops.torchvision.nms is not None)"
python -c "import torch; x = torch.randn(4096, 4096, device='cuda', dtype=torch.bfloat16); print('matmul ok', bool((x @ x).float().norm() > 0))"
python -c "import moonep; print('moonep', moonep.__file__)"
python -c "import cutlass; print('cutlass ok')"
python -c "import cuda.bindings.runtime as rt; print('cudart', rt.cudaGetDeviceCount())"
uv pip list 2>/dev/null | grep -iE "^torch|^triton|cutlass|moonep|attn|spmd|remat|cuda-bindings|cuda-python|^nvidia-nccl"
echo SETUP_VENV_DONE
