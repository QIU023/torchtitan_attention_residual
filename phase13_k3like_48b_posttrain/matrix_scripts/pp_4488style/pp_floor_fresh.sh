#!/bin/bash
# Noise-floor row: the same dp1 cell (bf16 flavor, 2x256) on a fresh triton + inductor cache, compared with the dp1 of the 4488-style run.
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export VENV=/workspace/venv_bfx9 SEED_ROOT=/workspace/.mx3_seeds_pprt512 SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up MEASURE_STEPS=5 WARM_STEPS=1
export CUDA_VISIBLE_DEVICES=2 TRITON_CACHE_DIR=/workspace/.triton_pprt512_fresh
D="--parallelism.data_parallel_shard_degree"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256"
TITAN=/tmp/wt_pprt CFG=kimi_k3_debugmodel BATCH="$B" CELLS="dp1|1|$D 1 $PD" $MX pprt512c
O1=$(ls -td /workspace/mx3_pprt512_* | head -1); O2=$(ls -td /workspace/mx3_pprt512c_* | head -1)
python3 - "$O1/dp1_measure.log" "$O2/dp1_measure.log" <<'PY'
import re, sys
def series(f):
    L, G = {}, {}
    for line in open(f, errors="ignore"):
        m = re.search(r"step: *(\d+)\s.*loss: *([0-9.]+).*grad_norm: *([0-9.]+)", line)
        if m: L[int(m.group(1))] = float(m.group(2)); G[int(m.group(1))] = float(m.group(3))
    return L, G
L1, G1 = series(sys.argv[1]); L2, G2 = series(sys.argv[2])
rows = [(s, L1[s], L2[s], abs(L2[s]-L1[s])/L1[s], G1[s], G2[s], abs(G2[s]-G1[s])/G1[s]) for s in sorted(L1) if s in L2 and s in G1 and s in G2]
for r in rows: print(f"step {r[0]}: dp1 cache A {r[1]:.6f} vs dp1 cache B {r[2]:.6f} (rel {r[3]:.2e}) | grad_norm {r[4]} vs {r[5]} (rel {r[6]:.2e})")
if rows: print(f"max relative loss diff {max(r[3] for r in rows):.2e}, max relative grad-norm diff {max(r[6] for r in rows):.2e}")
PY
echo "PP-FRESH-DONE"
