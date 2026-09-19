set -x
cd /workspace
until grep -q SETUP_STAGE1_DONE /workspace/setup_env.log; do sleep 15; done
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/bin/activate k3
pip install --no-cache-dir "grain==0.2.18" "datasets>=3.6.0,<4.8.0" tensorboard wandb "tyro>=1.0.5" "tokenizers>=0.15.0" safetensors einops pillow "spmd_types==0.2.5" av "torch_remat @ git+https://github.com/meta-pytorch/remat.git@d302699b1c58f83fa2c7b03bc2593967e9530335" "nvidia-cutlass-dsl==4.4.2" ninja pytest > /workspace/pip_deps.log 2>&1; echo "DEPS_RC=$?"
pip install --no-cache-dir --no-deps -e "/workspace/attn_gym[linear]" > /workspace/pip_attn_gym.log 2>&1 || pip install --no-cache-dir --no-deps -e /workspace/attn_gym >> /workspace/pip_attn_gym.log 2>&1; echo "AG_RC=$?"
pip install --no-cache-dir triton 2>&1 | tail -1
python -c "import torch, attn_gym, spmd_types, torch_remat, grain, tyro; print(IMPORTS_OK, torch.__version__)"
python /workspace/moonep_multicast_probe.py; echo "PROBE_RC=$?"
cd /workspace/moonep && git log --oneline -1
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=48
time pip install --no-build-isolation --no-deps -e . > /workspace/moonep_build.log 2>&1; echo "MOONEP_BUILD_RC=$?"; tail -3 /workspace/moonep_build.log
cd / && python -c "import moonep, torch; from moonep import Buffer, MoonEPCommPlan; print(MOONEP_OK, moonep.__file__)"; echo "MOONEP_IMPORT_RC=$?"
echo "SETUP_STAGE2_DONE"
