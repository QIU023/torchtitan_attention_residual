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
    git checkout 33327eb            # the public release of 2026-09-20
    TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=32 uv pip install -e . --no-build-isolation

The public release needs no source patch and pins `nvidia-cutlass-dsl==4.6.2`, which
is also what attn-gym's KDA needs, so the whole family goes to 4.6.2 and the version
conflict below disappears. The earlier pin `2bd860b` needs `cute.make_fragment`
replaced by `cute.make_rmem_tensor` in `moonep/grad_reduce.py` and the family at
4.6.0.

Checks that prove the stack: `pytest tests/unit_tests/gpu/test_kimi_k3_moon_ep.py` gives 2 passed, and a two-GPU debug step (`torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --training.steps 1`) reaches loss 12.58762.

Traps met on the way: `uv pip install -e .` fails for MoonEP without `--no-build-isolation` (its setup.py imports torch); a mixed cutlass family fails as an MLIR verification error inside MoonEP's planning kernel rather than as a version error; cutlass 4.4.2 breaks attn-gym's KDA with `module 'cutlass' has no attribute 'Vector'`; an apostrophe inside a single-quoted ssh payload closes the quote and the heredoc runs locally, so send scripts with scp.

## Traps, in the order they cost time on 2026-09-22

- **A mixed cutlass family reads as a kernel bug.** `nvidia-cutlass-dsl==X` does not pin `libs-base`, `libs-core`, `libs-cu12` or `libs-cu13`. With a cu130 torch, pip left the cu13 backend at 4.8.0 under a 4.5.1 front end and MoonEP's planning kernel failed MLIR verification with `nvvm.mbarrier.arrive.expect_tx ... is not a valid CombiningKind`. Two hours went into trying front-end versions before the backend versions were printed. Print every `nvidia-cutlass-dsl*` version after any install.
- **`uv pip install -e .` fails for MoonEP without `--no-build-isolation`**, since its setup.py imports torch; the message is `ModuleNotFoundError: No module named 'torch'` from inside the build.
- **`git checkout <rev>` fails silently when the working tree is dirty.** A `2>/dev/null` hid `error: Your local changes would be overwritten`, and the next twenty minutes were spent reading the old revision's sources while believing they were the new release's. Read another revision with `git archive <rev> | tar -x -C <dir>`, and check `git rev-parse HEAD` after any checkout.
- **An apostrophe inside a single-quoted ssh payload closes the quote**, and the rest of the heredoc runs on the local machine. Send scripts with scp.
- **`torchrun --master_port=2965$RANDOM`** can exceed 65535; use `$((29000+RANDOM%2000))`.
- **`.sum().backward()` is not a valid probe of a grouped GEMM**: its expanded ones tensor has stride 0 and the op rejects it, which looks like an empty-group bug and is not one. Pass `torch.randn_like(out)`.
- **The cutlass version swap is not free**: 4.4.2 runs MoonEP but breaks attn-gym's KDA with `module 'cutlass' has no attribute 'Vector'`; below 4.6 there is no combination that runs both, which is why the public release's 4.6.2 pin matters.
