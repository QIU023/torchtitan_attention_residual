#!/bin/bash
# H100 env smoke: torch cu126 nightly, and a 2-GPU NCCL all-reduce
. /workspace/kit_setup/env_cu126.sh
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available(), "devices", torch.cuda.device_count())
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
print("matmul ok", float((x @ x).float().abs().mean()))
import triton, spmd_types, torch_remat, attn_gym
print("triton", triton.__version__, "attn_gym ok")
PY
cat > /tmp/nccl_smoke.py <<'PY'
import os, torch, torch.distributed as dist
dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(r)
t = torch.ones(1, device="cuda") * (r + 1); dist.all_reduce(t)
print(f"rank {r} all_reduce {t.item()}"); dist.destroy_process_group()
PY
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 /tmp/nccl_smoke.py 2>&1 | grep -E "rank|Error" | head -4
