#!/bin/bash
# 10-19: main's K3 PP stage needs torch's _make_forward_chunk_state (10-03 nightly on), the last cu126 nightly is 09-07 and
# cu130 fails on driver 560, so build the 2026-10-05 nightly source (the 5060's torch, git 7bf52b3a) with the box's CUDA 12.6,
# sm90, sources in /dev/shm (noexec, so the build directories are symlinks to /workspace); then torchvision from source
# against it; venv_src = those wheels + the venv_cu126 package set. BUILD_OK only when every step succeeded.
exec > /workspace/build_torch.log 2>&1
set -x
fail() { echo "BUILD_FAILED $*"; exit 1; }
mkdir -p /dev/shm/tmp /dev/shm/uvcache
# Compiler temporaries and the uv cache are data files, fine on the noexec RAM disk; the 32 GB disk keeps only what runs.
export TMPDIR=/dev/shm/tmp UV_CACHE_DIR=/dev/shm/uvcache CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=176
B=/dev/shm/build; mkdir -p $B; cd $B
[ -d pt ] || git clone -q --depth 30 -b nightly https://github.com/pytorch/pytorch.git pt
cd pt
C=$(git log --format="%H %s" | grep "2026-10-05 nightly" | head -1 | cut -d' ' -f1); echo "commit $C"; [ -n "$C" ] || fail no-commit
git checkout -q $C && git log -1 --format="%H %s"
git submodule sync -q; git submodule update -q --init --recursive --depth 1 --jobs 32
[ -d build ] && [ ! -L build ] && mv build build_failed_noexec
mkdir -p /workspace/pt_build && ln -sfn /workspace/pt_build build && ls -la build
uv venv /workspace/venv_src --python 3.12; . /workspace/venv_src/bin/activate
uv pip install -r requirements.txt build ninja cmake
# NCCL's own makefile runs a script from the noexec source tree, so link the pip NCCL 2.29.3 (the version torch pins).
export USE_CUDA=1 USE_ROCM=0 USE_XPU=0 BUILD_TEST=0 USE_CUSPARSELT=0 USE_CUDSS=0 USE_NVSHMEM=0 USE_MPI=0 USE_KINETO=1 CMAKE_BUILD_TYPE=Release \
  USE_SYSTEM_NCCL=1 NCCL_ROOT=/workspace/nccl_sys NCCL_INCLUDE_DIR=/workspace/nccl_sys/include NCCL_LIB_DIR=/workspace/nccl_sys/lib RERUN_CMAKE=1
export LD_LIBRARY_PATH=/workspace/nccl_sys/lib:$LD_LIBRARY_PATH
python -m build --wheel --no-isolation > /workspace/pt_build_inner.log 2>&1 || fail torch
ls -la dist/
uv pip install dist/torch-*.whl || fail torch-install
(cd /workspace && python -c "import torch; print('built torch', torch.__version__, torch.version.git_version, torch.version.cuda, torch.cuda.is_available())") || fail torch-import
cd $B; [ -d vision ] || git clone -q --depth 200 https://github.com/pytorch/vision.git vision
cd vision; VC=$(git log --before="2026-10-06" --format=%H -1); git checkout -q $VC && echo "vision $VC"
mkdir -p /workspace/vision_build && ln -sfn /workspace/vision_build build
FORCE_CUDA=1 python setup.py bdist_wheel > /workspace/vision_build.log 2>&1 || fail vision
uv pip install --no-deps dist/torchvision-*.whl || fail vision-install
UV_HTTP_TIMEOUT=900 uv pip install -r /workspace/kit_setup/freeze_rest.txt "cuda-bindings>=12.9,<13" "cuda-python>=12.9,<13"
uv pip install "attn-gym[linear]==0.0.17" \
  "spmd_types @ git+https://github.com/meta-pytorch/spmd_types.git@fce1acb1e9bbce66f398893ff7836346f4cf03cd" \
  "torch_remat @ git+https://github.com/meta-pytorch/remat.git@624f19cbdb70720bd05988d6485a1225f46c074b" \
  "torch_checkpointing @ git+https://github.com/meta-pytorch/torch_checkpointing.git@dcbb3a59cfad1cbeb95994f0d6cb1802c1d91825"
TV=$(python - <<'PY'
import re
t = open("/dev/shm/build/pt/.ci/docker/triton_version.txt").read().strip()
c = open("/dev/shm/build/pt/.ci/docker/ci_commit_pins/triton.txt").read().strip()[:8]
print(f"{t}+git{c}")
PY
)
cd /workspace
echo "triton pin $TV"; uv pip install --index-url https://download.pytorch.org/whl/nightly/ "triton==$TV" || fail triton
python - <<'PY'
import torch, torchvision, triton
print("venv_src", torch.__version__, torch.version.git_version[:10], "cuda", torch.version.cuda, "tv", torchvision.__version__, "triton", triton.__version__)
import torch.distributed.pipelining.stage as s
print("has _make_forward_chunk_state", hasattr(s._PipelineStageBase, "_make_forward_chunk_state"))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print("matmul", float((x @ x).float().abs().mean()))
PY
[ $? -eq 0 ] || fail smoke
df -h /workspace /dev/shm | tail -2
echo BUILD_OK
