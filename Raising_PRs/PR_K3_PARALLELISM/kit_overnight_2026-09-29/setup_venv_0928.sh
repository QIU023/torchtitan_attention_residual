#!/bin/bash
# T1a: the 09-28 cu130 nightly on the 5060 box, so main runs without the compat shim and torch allocates
# receive buffers just in time, as CI's torch does.
set -x
V=/workspace/venv_0928; T=/workspace/torchtitan_attention_residual/torchtitan
[ -d $V ] || uv venv $V --python 3.12
. $V/bin/activate
uv pip install --pre "torch==2.15.0.dev20260928+cu130" "torchvision==0.30.0.dev20260928+cu130" --index-url https://download.pytorch.org/whl/nightly/cu130
git -C $T show 5dc97a3e7:.ci/docker/requirements.txt > /tmp/req_5dc9.txt
git -C $T show 5dc97a3e7:.ci/docker/requirements-vlm.txt > /tmp/req_vlm_5dc9.txt
uv pip install -r /tmp/req_5dc9.txt -r /tmp/req_vlm_5dc9.txt
uv pip install pytest pytest-subtests expecttest pyflakes ufmt==2.8.0 black==24.10.0 usort==1.0.8.post1 mooncake-transfer-engine==0.3.13.post1
uv pip install --no-deps transformers
C=4.6.2
uv pip install nvidia-cutlass-dsl==$C nvidia-cutlass-dsl-libs-base==$C nvidia-cutlass-dsl-libs-core==$C nvidia-cutlass-dsl-libs-cu13==$C nvidia-cutlass-dsl-libs-cu12==$C
python -c "import torch, torchvision; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.device_count(), 'tv', torchvision.__version__)"
python -c "import torch.distributed.config as c; print('per_edge', hasattr(c, 'pipeline_per_edge_p2p'))"
python -c "import torch.distributed.pipelining._recv_buffers as r; print('recv_buffers ok')"
python -c "import inspect, torch.distributed.pipelining.schedules as s; print('unshard_lookahead', 'unshard_lookahead' in inspect.signature(s.ScheduleInterleaved1F1B.__init__).parameters)"
python -c "import torch; torch.backends.cuda.matmul.fp32_precision = 'bfx9'; print('bfx9 ok')"
uv pip list 2>/dev/null | grep -iE "^torch|^triton|cutlass|attn|mooncake|spmd|remat|cuda-bindings|transformers"
echo VENV_0928_DONE
