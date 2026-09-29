#!/bin/bash
# One-time setup on the H100 box: venv, torch nightly (cu130), the tree's requirements,
# the cutlass DSL family at MoonEP's pin, MoonEP 33327eb, and the fork at moonep_review1.
set -euxo pipefail
M=~/mep; mkdir -p $M; cd $M
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH
[ -d venv ] || uv venv venv --python 3.12
. venv/bin/activate
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
[ -d tt ] || git clone https://github.com/QIU023/torchtitan.git tt
git -C tt fetch -q origin moonep_review1
git -C tt checkout -q --detach FETCH_HEAD
git -C tt rev-parse --short HEAD
uv pip install -r tt/.ci/docker/requirements.txt -r tt/.ci/docker/requirements-vlm.txt
uv pip install pytest pytest-subtests expecttest
V=4.6.2
uv pip install nvidia-cutlass-dsl==$V nvidia-cutlass-dsl-libs-base==$V nvidia-cutlass-dsl-libs-core==$V \
  nvidia-cutlass-dsl-libs-cu13==$V nvidia-cutlass-dsl-libs-cu12==$V
[ -d MoonEP ] || git clone -q https://github.com/MoonshotAI/MoonEP.git
git -C MoonEP checkout -q 33327eb
test "$(git -C MoonEP rev-parse --short=7 HEAD)" = 33327eb
(cd MoonEP && TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 uv pip install -e . --no-build-isolation)
uv pip list 2>/dev/null | grep -iE "^torch |cutlass|attn-gym|moonep|spmd|remat"
# Every nvidia-cutlass-dsl* line above must read 4.6.2; a mixed family fails inside JIT kernels.
