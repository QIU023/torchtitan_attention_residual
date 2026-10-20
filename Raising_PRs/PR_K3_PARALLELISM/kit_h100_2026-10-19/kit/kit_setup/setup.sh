#!/bin/bash
# 10-19 H100 box 115.124.123.239:50752 (driver 560 + 12.8 compat, 32 GB disk): clone the trees, venv_cu126 = torch 0906 cu126
# + the 5060 package set + main's pins (attn-gym 0.0.17, spmd_types, torch_remat, torch_checkpointing), then the env smoke.
exec > /workspace/setup.log 2>&1
set -x
export UV_CACHE_DIR=/workspace/.uvcache
git clone -q --depth 1 -b k3_cp_mm https://github.com/QIU023/torchtitan.git /workspace/tt_cpmm && echo "tt_cpmm $(git -C /workspace/tt_cpmm rev-parse HEAD)"
git clone -q --depth 1 -b k3_pp_mm https://github.com/QIU023/torchtitan.git /workspace/tt_dep && echo "tt_dep $(git -C /workspace/tt_dep rev-parse HEAD)"
git clone -q --depth 120 https://github.com/pytorch/torchtitan.git /workspace/tt_main && git -C /workspace/tt_main checkout -q e61df55b6 && echo "tt_main $(git -C /workspace/tt_main rev-parse HEAD)"
command -v uv || pip install -q uv
uv venv --python 3.12 /workspace/venv_cu126 || exit 1
. /workspace/venv_cu126/bin/activate
uv pip install "torch==2.15.0.dev20260906+cu126" "torchvision==0.30.0.dev20260906+cu126" --index-url https://download.pytorch.org/whl/nightly/cu126 || { echo PIP_TORCH_FAILED; echo SETUP_DONE; exit 1; }
UV_HTTP_TIMEOUT=900 uv pip install -r /workspace/kit_setup/freeze_rest.txt "cuda-bindings>=12.9,<13" "cuda-python>=12.9,<13" && echo PIP_OK || echo PIP_REST_FAILED
UV_HTTP_TIMEOUT=900 uv pip install "attn-gym[linear]==0.0.17" \
  "spmd_types @ git+https://github.com/meta-pytorch/spmd_types.git@fce1acb1e9bbce66f398893ff7836346f4cf03cd" \
  "torch_remat @ git+https://github.com/meta-pytorch/remat.git@624f19cbdb70720bd05988d6485a1225f46c074b" \
  "torch_checkpointing @ git+https://github.com/meta-pytorch/torch_checkpointing.git@dcbb3a59cfad1cbeb95994f0d6cb1802c1d91825" && echo PINS_OK || echo PINS_FAILED
V=4.6.2
UV_HTTP_TIMEOUT=900 uv pip install nvidia-cutlass-dsl==$V nvidia-cutlass-dsl-libs-base==$V nvidia-cutlass-dsl-libs-core==$V nvidia-cutlass-dsl-libs-cu12==$V "cuda-bindings>=12.9,<13" "cuda-python>=12.9,<13" && echo CUTLASS_OK || echo CUTLASS_FAILED
python -c "import torch; print('torch', torch.__version__)"
df -h /workspace | tail -1
/workspace/kit_setup/smoke_env.sh
echo SETUP_DONE
