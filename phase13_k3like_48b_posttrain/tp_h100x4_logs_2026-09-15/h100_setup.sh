#!/bin/bash
# H100 setup for the PR 4499 round-3 rerun (kit: /workspace/kit_tp). Log: /workspace/setup.log
set -eo pipefail
cd /workspace
uv venv -q --python 3.12 /workspace/venv
source /workspace/venv/bin/activate
uv pip install -q --pre "torch==2.15.0.dev20260906+cu130" --index-url https://download.pytorch.org/whl/nightly/cu130
python -c "import torch; print('torch', torch.__version__, torch.version.cuda)"
[ -d tt ] || git clone -q -b tpsp_review4 https://github.com/QIU023/torchtitan.git tt
cd tt; echo "branch head: $(git log --oneline -1)"
git fetch -q https://github.com/pytorch/torchtitan.git main
[ -d ../tt_parent ] || git worktree add -q --detach ../tt_parent d34a13fdf
echo "parent head: $(git -C ../tt_parent log --oneline -1)"
uv pip install -q -r requirements.txt pytest==7.3.2 expecttest transformers numpy
python - <<'PY'
import torch, importlib.metadata as md
print("after deps: torch", torch.__version__)
for p in ("triton", "spmd_types", "attn-gym", "torch_remat", "grain", "tyro"):
    try: print(p, md.version(p))
    except Exception as e: print(p, "missing", e)
PY
case "$(python -c 'import torch;print(torch.__version__)')" in 2.15.0.dev20260906+cu130) ;; *) echo "TORCH REPLACED, reinstalling"; uv pip install -q --pre --no-deps --reinstall "torch==2.15.0.dev20260906+cu130" --index-url https://download.pytorch.org/whl/nightly/cu130 ;; esac
python /workspace/kit_tp/hacks/kda_capability_hack.py /workspace/tt
python /workspace/kit_tp/hacks/kda_capability_hack.py /workspace/tt_parent
python - <<'PY'
import torch
print("gpus", torch.cuda.device_count(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
try:
    from torch._C._distributed_c10d import _SymmetricMemory
    print("multicast", [_SymmetricMemory.has_multicast_support(torch._C._autograd.DeviceType.CUDA, i) for i in range(torch.cuda.device_count())])
except Exception as e:
    print("multicast check via SymmetricMemory failed:", repr(e))
try:
    import ctypes
    cu = ctypes.CDLL("libcuda.so.1"); cu.cuInit(0); v = ctypes.c_int()
    for i in range(torch.cuda.device_count()):
        d = ctypes.c_int(); cu.cuDeviceGet(ctypes.byref(d), i)
        r = cu.cuDeviceGetAttribute(ctypes.byref(v), 132, d)  # CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED
        print(f"cuda attr multicast dev{i}: rc={r} value={v.value}")
except Exception as e:
    print("libcuda attr check failed:", repr(e))
PY
echo SETUP-DONE
