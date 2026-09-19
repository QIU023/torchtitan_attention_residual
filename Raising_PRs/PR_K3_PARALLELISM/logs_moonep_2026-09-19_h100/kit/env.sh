# 4 x H100 box (115.124.123.240:16923), 2026-09-19.
# uv venv with torch 2.15.0.dev20260907+cu126 against the cuda-toolkit-12-6 stage 1 installs.
export VIRTUAL_ENV=/workspace/venv_k3
export PATH=/workspace/venv_k3/bin:/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
export NCCL_NVLS_ENABLE=0 MOONEP_MEM_HANDLE_TYPE=${MOONEP_MEM_HANDLE_TYPE:-auto}
export TITAN=/workspace/src/tt_moonep PYTHONPATH=/workspace/src/tt_moonep
export TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HOME=/workspace/.hf_home
export TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor TRITON_CACHE_DIR=/workspace/.triton
mkdir -p /workspace/results /workspace/.hf_home
# cutlass-dsl 4.6.0 pulls cuda-python 13.x, whose bindings ask for a CUDA 13
# driver; this box's 560.35.03 admits 12.8, so cudaGetDeviceCount returns
# cudaErrorInsufficientDriver and every MoonEP suite fails before it starts.
# Pinned back with: uv pip install "cuda-python<13" "cuda-bindings<13"
