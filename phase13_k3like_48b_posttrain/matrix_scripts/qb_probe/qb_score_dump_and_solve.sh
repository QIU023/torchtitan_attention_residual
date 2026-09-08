#!/bin/bash
# Dump two steps of router scores from the frozen debug model (tip tree), then solve QB offline on the real batches.
cd /tmp/wt_qbtip; source /workspace/venv_bfx9/bin/activate; export PYTHONPATH=/tmp/attn_gym_up:/tmp/wt_qbtip
rm -rf /workspace/qb_scores; QB_PROBE_SCORE_DIR=/workspace/qb_scores QB_PROBE_ROUTE_STEPS=1,2 CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=/workspace/.inductor_qbscores timeout 1500 torchrun --nproc_per_node=1 --master_port=29931 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_probe_frozen --debug.seed 42 --debug.deterministic --training.steps 2 --metrics.log_freq 1 --training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.spmd_backend partial_dtensor > /workspace/qb_scores_run.log 2>&1
echo "dump rc=$? files: $(ls /workspace/qb_scores 2>/dev/null | tr '\n' ' ')"
python - <<'PY'
import sys, torch
sys.path.insert(0, "/tmp/wt_qbtip")
from torchtitan.components.quantile_balance import topk_with_cutoff, margin_histogram, quantile_balance_bias_histogram, quantile_balance_bias, expert_loads
A = torch.load("/workspace/qb_scores/scores_step1_rank0.pt"); B = torch.load("/workspace/qb_scores/scores_step2_rank0.pt")
k = 4
def cv(scores, bias):
    l = expert_loads(scores, bias, k).float(); return (l.std(unbiased=False) / l.mean()).item()
def solve(scores, bias, mode, bins=512):
    _, cutoff = topk_with_cutoff(scores, bias, k)
    if mode == "exact": return quantile_balance_bias(scores, cutoff, k)
    if mode == "fixed": lo, hi = -1.0, 1.0
    else: bmin, bmax = bias.min().item(), bias.max().item(); lo, hi = -(1.0 + bmax), 1.0 - bmin
    return quantile_balance_bias_histogram(margin_histogram(scores, cutoff, num_bins=bins, lo=lo, hi=hi), k, lo=lo, hi=hi)
layers = sorted({key[1] for key in A})
print(f"layers {len(layers)}, tokens per layer {A[(1, layers[0])].shape[0]}, experts {A[(1, layers[0])].shape[1]}")
sa = A[(1, layers[0])]; print(f"layer {layers[0]} score stats: mean {sa.mean():.4f} std {sa.std():.4f}; per-token spread (max-min over experts) mean {(sa.max(1).values - sa.min(1).values).mean():.4f}; ties: mean number of experts within 1e-3 of the 5th-largest score per token {((sa - sa.topk(5, dim=1).values[:, -1:]).abs() < 1e-3).sum(1).float().mean():.2f}")
for mode, bins in (("exact", 0), ("fixed", 512), ("fixed", 4096), ("adaptive", 1000)):
    same, xfer = [], []
    for L in layers:
        s1, s2 = A[(1, L)], B[(2, L)]
        bias = torch.zeros(s1.shape[1])
        traj = []
        for it in range(10):
            bias = solve(s1, bias, mode, bins); traj.append(cv(s1, bias))
        same.append(traj); xfer.append(cv(s2, bias))
    t = torch.tensor(same)
    print(f"{mode:8s} bins={bins:4d}: same-batch cv after 1/2/3/5/10 solves (mean over layers) = {t[:,0].mean():.3f} / {t[:,1].mean():.3f} / {t[:,2].mean():.3f} / {t[:,4].mean():.3f} / {t[:,9].mean():.3f}; that bias applied to the NEXT batch: cv {torch.tensor(xfer).mean():.3f}")
# one solve from zero on batch 1, applied to batch 2 (what the live run does at step 2)
one = []
for L in layers:
    bias = solve(A[(1, L)], torch.zeros(A[(1, L)].shape[1]), "fixed", 512); one.append((cv(A[(1, L)], torch.zeros(32)), cv(A[(1, L)], bias), cv(B[(2, L)], bias), cv(B[(2, L)], torch.zeros(32))))
o = torch.tensor(one); print(f"live-run step 2 replica: batch-1 cv before/after one histogram solve {o[:,0].mean():.3f} -> {o[:,1].mean():.3f}; that bias on batch 2: {o[:,2].mean():.3f} (batch 2 unbiased: {o[:,3].mean():.3f})")
print("SOLVE-DONE")
PY
