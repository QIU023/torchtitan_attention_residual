#!/bin/bash
# One-time setup on the 4 x H100 box (driver 560 + compat 12.8, so torch nightly cu128):
# venv, torch, the tree's requirements, the cutlass DSL family at MoonEP's pin, MoonEP 33327eb,
# and one worktree per head the H100 plan measures.
set -uxo pipefail
M=~/mep; mkdir -p $M; cd $M
[ -d venv ] || uv venv venv --python 3.12
. venv/bin/activate
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
[ -d tt ] || git clone -q https://github.com/QIU023/torchtitan.git tt
cd tt
git remote add upstream https://github.com/pytorch/torchtitan.git 2>/dev/null
git fetch -q upstream main
git fetch -q origin k3_ac_reuse_attention pp_review_optimize pp_offload_review1 pp_balance_review1 k3_pp_mm moonep_review1
declare -A HEADS=([main_f359]=f35966713 [main_5dc9]=5dc97a3e7 [pr4656]=5d469fdf3 [pra]=85eefa54b
  [dev]=1777ad806 [o4765]=56f4cd3b0 [b4764]=0a9034257 [dep]=a03f74981 [moonep]=a505f74a8)
for n in "${!HEADS[@]}"; do
  [ -d $M/w/$n ] || git worktree add -q --detach $M/w/$n ${HEADS[$n]}
  echo "$n $(git -C $M/w/$n rev-parse --short HEAD)"
done
cd $M
uv pip install -r w/main_5dc9/.ci/docker/requirements.txt -r w/main_5dc9/.ci/docker/requirements-vlm.txt
uv pip install pytest pytest-subtests expecttest
V=4.6.2
uv pip install nvidia-cutlass-dsl==$V nvidia-cutlass-dsl-libs-base==$V nvidia-cutlass-dsl-libs-core==$V \
  nvidia-cutlass-dsl-libs-cu13==$V nvidia-cutlass-dsl-libs-cu12==$V
[ -d MoonEP ] || git clone -q https://github.com/MoonshotAI/MoonEP.git
git -C MoonEP checkout -q 33327eb
(cd MoonEP && TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 uv pip install -e . --no-build-isolation)
python -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count())"
python -c "import moonep; print('moonep ok')"
uv pip list 2>/dev/null | grep -iE "^torch |cutlass|attn-gym|moonep|spmd|remat"
echo SETUP_DONE
