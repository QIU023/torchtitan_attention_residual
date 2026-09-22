# MoonEP on an H100 / H200 NVSwitch box: the environment that works

Verified 2026-09-22 on 4 x H100 80GB HBM3 (NV18, multicast 1), driver 580.126.20, CUDA 13.0, Ubuntu user `ubuntu`.

Gate first, before installing anything:

    python3 -c "import ctypes;c=ctypes.CDLL('libcuda.so.1');c.cuInit(0);v=ctypes.c_int();c.cuDeviceGetAttribute(ctypes.byref(v),132,0);print('multicast',v.value)"

`multicast 1` is required: MoonEP's Buffer allocates a multicast meta buffer and asserts switch multicast, which a peer-to-peer NVLink box does not have.

    curl -LsSf https://astral.sh/uv/install.sh | sh
    uv venv --python 3.12 ~/venv_k3 && source ~/venv_k3/bin/activate
    uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
    git clone https://github.com/QIU023/torchtitan.git ~/tt && cd ~/tt && git checkout k3_moonep_seam
    uv pip install -r .ci/docker/requirements.txt
    uv pip install renderers==0.1.11 apache-tvm-ffi pytest pytest-subtests expecttest
    # the whole cutlass family at one version: pip otherwise leaves the cu13 backend at 4.8.0
    uv pip install nvidia-cutlass-dsl==4.6.0 nvidia-cutlass-dsl-libs-base==4.6.0 \
      nvidia-cutlass-dsl-libs-core==4.6.0 nvidia-cutlass-dsl-libs-cu13==4.6.0 \
      nvidia-cutlass-dsl-libs-cu12==4.6.0
    git apply matrix_scripts/local_hacks/kda_sm120_guard_lift.patch   # KDA refuses anything but Blackwell without it
    git clone https://github.com/MoonshotAI/MoonEP.git ~/MoonEP && cd ~/MoonEP
    git checkout 2bd860b4dd083df62b79d5e916fca71ec5742228
    sed -i 's/cute\.make_fragment/cute.make_rmem_tensor/g' moonep/grad_reduce.py
    TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 uv pip install -e . --no-build-isolation

Checks that prove the stack: `pytest tests/unit_tests/gpu/test_kimi_k3_moon_ep.py` gives 2 passed, and a two-GPU debug step (`torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --training.steps 1`) reaches loss 12.58762.

Traps met on the way: `uv pip install -e .` fails for MoonEP without `--no-build-isolation` (its setup.py imports torch); a mixed cutlass family fails as an MLIR verification error inside MoonEP's planning kernel rather than as a version error; cutlass 4.4.2 breaks attn-gym's KDA with `module 'cutlass' has no attribute 'Vector'`; an apostrophe inside a single-quoted ssh payload closes the quote and the heredoc runs locally, so send scripts with scp.
