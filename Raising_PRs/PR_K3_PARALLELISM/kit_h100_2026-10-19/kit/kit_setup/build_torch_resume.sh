#!/bin/bash
# Resume of build_torch.sh after its wheel is built and installed: the import check onward.
exec > /workspace/build_torch.log 2>&1
set -x
fail() { echo "BUILD_FAILED $*"; exit 1; }
mkdir -p /dev/shm/tmp /dev/shm/uvcache
# Compiler temporaries and the uv cache are data files, fine on the noexec RAM disk; the 32 GB disk keeps only what runs.
export TMPDIR=/dev/shm/tmp UV_CACHE_DIR=/dev/shm/uvcache CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=176
B=/dev/shm/build; . /workspace/venv_src/bin/activate
export LD_LIBRARY_PATH=/workspace/nccl_sys/lib:$LD_LIBRARY_PATH
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
