#!/bin/bash
# Attribute the residual QAT rollout gap: weights-only fake-quant forward vs the full (weights + activations) one.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
source /workspace/venv_verl/bin/activate; export PYTHONPATH=/tmp/wt_k3int_cp:/tmp/attn_gym_up HF_HOME=/workspace/.hf_home TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_rescore TRITON_CACHE_DIR=/workspace/.triton_rescore
CUDA_VISIBLE_DEVICES=${GPU:-0} python $S/rescore_tt_qat_wonly.py 2>&1 | grep -v "W09\|WARNING\|deprecat" | tail -1
CUDA_VISIBLE_DEVICES=${GPU:-0} python - <<'PY' 2>&1 | grep -v "W09\|Warning"
import torch
D = "/workspace/logprob_dump2"; tt = torch.load(f"{D}/rescore_tt.pt"); q = torch.load(f"{D}/rescore_tt_qat.pt"); w = torch.load(f"{D}/rescore_tt_qat_wonly.pt")
def stat(name, pairs):
    d = torch.cat([(x - y).abs() for x, y in pairs]); print(f"{name:58s} mean|d|={d.mean():.4f} frac>0.5={(d > 0.5).float().mean():.3f}")
idx = sorted(tt)
stat("weights-only fake-quant forward vs bf16 forward (weight quant error)", [(w[i]["lp"], tt[i]["lp"]) for i in idx])
stat("full fake-quant forward vs weights-only (activation quant error)", [(q[i]["lp"], w[i]["lp"]) for i in idx])
print("ATTRIBUTION-DONE")
PY
