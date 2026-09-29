#!/bin/bash
# setup_src.sh let torchvision's install resolve "torch" to PyPI's 2.11.0+cu130 and a mixed cutlass family;
# put the source-built torch and the pinned Triton back without dependency resolution, drop the cu13
# runtime packages torch 2.11 brought, and build MoonEP without letting it resolve torch.
set -x
M=~/mep; . $M/venv_src/bin/activate; cd $M
uv pip install --reinstall --no-deps $M/pt/dist/torch-*.whl
uv pip install --reinstall --no-deps --pre "triton==3.8.0+gitc01b6774" --index-url https://download.pytorch.org/whl/nightly/
uv pip uninstall cuda-toolkit nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 \
  nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 \
  nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx
uv pip install --reinstall --no-deps nvidia-cutlass-dsl-libs-cu13==4.6.2
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.nccl.version())"
python -c "import torch.distributed.config as c; print('per_edge', hasattr(c, 'pipeline_per_edge_p2p'))"
python -c "import inspect, torch.distributed.pipelining.schedules as s; print('unshard_lookahead', 'unshard_lookahead' in inspect.signature(s.ScheduleInterleaved1F1B.__init__).parameters)"
python -c "import triton; print('triton', triton.__version__)"
python -c "import torchvision, torch; print('torchvision', torchvision.__version__, torch.ops.torchvision.nms is not None)"
python -c "import torch; x = torch.randn(2048, 2048, device='cuda'); print('matmul', (x @ x).float().norm().item() > 0)"
(cd MoonEP_src && TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 CUDA_HOME=/usr/local/cuda-12.8 PATH=/usr/local/cuda-12.8/bin:$PATH \
  uv pip install -e . --no-build-isolation --no-deps) > $M/moonep_build_src.log 2>&1
echo "moonep rc=$?"
python -c "import moonep; print('moonep', moonep.__file__)"
python -c "from torchtitan.distributed.activation_storage import RemoteBackend" 2>/dev/null; python -c "from mooncake.engine import TransferEngine; print('mooncake ok')"
uv pip list 2>/dev/null | grep -iE "^torch|^triton|cutlass|moonep|mooncake|cuda-bindings|cuda-python|^nvidia-nccl|^nvidia-cuda-runtime"
echo FIX_SRC_DONE
