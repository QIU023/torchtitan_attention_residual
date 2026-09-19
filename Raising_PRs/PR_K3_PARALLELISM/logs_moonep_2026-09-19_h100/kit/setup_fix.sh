set -x
export PATH=/root/miniconda3/bin:$PATH
source /root/miniconda3/bin/activate k3
cd /workspace
pip uninstall -y torch torchvision triton > /dev/null 2>&1
pip install --no-cache-dir "torch==2.15.0.dev20260906+cu126" "torchvision==0.30.0.dev20260906+cu126" --index-url https://download.pytorch.org/whl/nightly/cu126 > /workspace/pip_torch2.log 2>&1; echo "TORCH_RC=$?"; tail -1 /workspace/pip_torch2.log | cut -c1-200
python -c "import torch; print(TORCH, torch.__version__, torch.version.cuda, torch.cuda.device_count())"
rm -rf /workspace/attn_gym_src && mkdir -p /workspace/attn_gym_src && tar -C /workspace/attn_gym_src --strip-components=1 -xzf /workspace/attn_gym_src.tgz
pip uninstall -y attn-gym attn_gym > /dev/null 2>&1
pip install --no-cache-dir --no-deps -e /workspace/attn_gym_src > /workspace/pip_attn_gym2.log 2>&1; echo "AG_RC=$?"
python -c "import torch, attn_gym, spmd_types, torch_remat, grain, tyro, einops, datasets; from attn_gym.linear.kda import chunk_kda; print(IMPORTS_OK, torch.__version__, attn_gym.__file__)"; echo "IMPORTS_RC=$?"
python /workspace/moonep_multicast_probe.py; echo "PROBE_RC=$?"
cd /workspace/moonep && git log --oneline -1
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=48
time pip install --no-build-isolation --no-deps -e . > /workspace/moonep_build2.log 2>&1; echo "MOONEP_BUILD_RC=$?"; tail -3 /workspace/moonep_build2.log | cut -c1-200
cd / && python -c "import moonep, torch; from moonep import Buffer, MoonEPCommPlan; print(MOONEP_OK, moonep.__file__)"; echo "MOONEP_IMPORT_RC=$?"
echo "SETUP_FIX_DONE"
