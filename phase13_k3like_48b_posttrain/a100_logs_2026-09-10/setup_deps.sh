#!/bin/bash
until grep -q TORCH-DONE /workspace/setup_torch.log 2>/dev/null; do sleep 15; done
source /workspace/venv/bin/activate
cd /workspace/wt_tp
uv pip install -r requirements.txt 2>&1 | tail -2
uv pip install --no-deps torchao==0.18.0 2>&1 | tail -1
uv pip install spmd_types==0.2.5 2>&1 | tail -1 || echo "spmd_types pypi failed"
python -c "import spmd_types" 2>/dev/null || (cp -r /workspace/spmd_types_pkg /workspace/venv/lib/python3.12/site-packages/spmd_types && echo "spmd_types copied")
uv pip install pytest 2>&1 | tail -1
python -c "import torch, spmd_types, torchao; print(\"torch\", torch.__version__, torch.version.cuda, \"cuda ok\", torch.cuda.is_available(), torch.cuda.device_count())"
PYTHONPATH=/workspace/attn_gym_up:/workspace/wt_tp python -c "import attn_gym; from attn_gym.linear.kda import chunk_kda; print(\"attn_gym ok\")"
echo DEPS-DONE
