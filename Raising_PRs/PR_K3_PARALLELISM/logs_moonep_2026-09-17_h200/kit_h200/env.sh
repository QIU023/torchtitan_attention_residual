# H200 box (217.18.55.111), 2026-09-17: uv venv with torch 2.15.0.dev20260906+cu126, nvcc 12.6.
export VIRTUAL_ENV=/workspace/venv_k3
export PATH=/workspace/venv_k3/bin:/usr/local/cuda-12.6/bin:$PATH
export CUDA_HOME=/usr/local/cuda-12.6
export NCCL_NVLS_ENABLE=0 MOONEP_MEM_HANDLE_TYPE=${MOONEP_MEM_HANDLE_TYPE:-auto}
export TITAN=/workspace/tt_moonep PYTHONPATH=/workspace/tt_moonep
export TORCHINDUCTOR_COMPILE_THREADS=1
export HF_HOME=/workspace/.hf_home
