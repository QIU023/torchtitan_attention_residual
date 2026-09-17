set -x
UV=/root/miniconda3/envs/py3.10/bin/uv
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH
cd /workspace
$UV pip install "torch==2.15.0.dev20260906+cu126" "torchvision==0.30.0.dev20260906+cu126" --index-url https://download.pytorch.org/whl/nightly/cu126 > /workspace/uv_torch.log 2>&1; echo "TORCH_RC=$?"; tail -1 /workspace/uv_torch.log | cut -c1-200
python -c "import torch; print(TORCH, torch.__version__, torch.version.cuda, torch.cuda.device_count())"
$UV pip install "grain==0.2.18" "datasets>=3.6.0,<4.8.0" tensorboard wandb "tyro>=1.0.5" "tokenizers>=0.15.0" safetensors einops pillow "spmd_types==0.2.5" av "torch_remat @ git+https://github.com/meta-pytorch/remat.git@d302699b1c58f83fa2c7b03bc2593967e9530335" "nvidia-cutlass-dsl==4.4.2" ninja pytest setuptools wheel pip > /workspace/uv_deps.log 2>&1; echo "DEPS_RC=$?"; tail -1 /workspace/uv_deps.log | cut -c1-200
rm -rf /workspace/attn_gym_src && mkdir -p /workspace/attn_gym_src && tar -C /workspace/attn_gym_src --strip-components=1 -xzf /workspace/attn_gym_src.tgz
$UV pip install --no-deps -e /workspace/attn_gym_src > /workspace/uv_attn_gym.log 2>&1; echo "AG_RC=$?"; tail -1 /workspace/uv_attn_gym.log | cut -c1-200
python -c "import torch; print(torch after deps:, torch.__version__)"
python -c "import torch, attn_gym, spmd_types, torch_remat, grain, tyro, einops, datasets; from attn_gym.linear.kda import chunk_kda; print(IMPORTS_OK, torch.__version__, attn_gym.__file__)"; echo "IMPORTS_RC=$?"
python /workspace/moonep_multicast_probe.py; echo "PROBE_RC=$?"
cd /workspace/moonep && git log --oneline -1
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=48
time python -m pip install --no-build-isolation --no-deps -e . > /workspace/moonep_build3.log 2>&1; echo "MOONEP_BUILD_RC=$?"; tail -3 /workspace/moonep_build3.log | cut -c1-200
cd / && python -c "import moonep, torch; from moonep import Buffer, MoonEPCommPlan; print(MOONEP_OK, moonep.__file__)"; echo "MOONEP_IMPORT_RC=$?"
echo "SETUP_UV_DONE"
