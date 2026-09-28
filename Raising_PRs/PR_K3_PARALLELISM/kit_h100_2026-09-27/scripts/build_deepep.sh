#!/bin/bash
# DeepEP v2 at the commit torchtitan's CI installs, into venv_k3 (torch 0921 cu130), for the MoonEP comparison.
set -x
export PATH=$HOME/.local/bin:/usr/local/cuda/bin:$PATH
source ~/venv_k3/bin/activate
sudo -n apt-get install -y -qq rdma-core libibverbs1 libmlx5-1 libibverbs-dev || echo "apt failed"
D=~/DeepEP_v2
[ -d $D ] || git clone -q --recursive https://github.com/deepseek-ai/DeepEP.git $D
git -C $D checkout -q 01dc3aaac82068020353dce2c302e38153c0bfaa && git -C $D submodule update -q --init --recursive
git -C $D rev-parse --short HEAD
cd $D && CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=24 uv pip install --no-build-isolation . 2>&1 | tail -20
python -c "from deep_ep import ElasticBuffer; print('deep_ep ok')"
echo BUILD_DONE
