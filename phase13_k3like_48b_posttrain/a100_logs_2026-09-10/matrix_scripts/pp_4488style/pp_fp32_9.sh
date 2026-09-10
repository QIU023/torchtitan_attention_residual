#!/bin/bash
# fp32 masters (training.dtype float32), 9-layer local alias so the 1-GPU cell fits 16 GB; 1 GPU vs 2-GPU PP, 5 steps, two 256-token micro-batches.
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export VENV=/workspace/venv_bfx9 SEED_ROOT=/workspace/.mx3_seeds_pprt_fp32l9 SEED_CFG=kimi_k3_debugmodel9 PYPRE=/tmp/attn_gym_up MEASURE_STEPS=5 WARM_STEPS=1
export SEED_EXTRA="--training.dtype float32" KEYSEED=fp32
export CUDA_VISIBLE_DEVICES=0,1 TRITON_CACHE_DIR=/workspace/.triton_pprt_fp32
D="--parallelism.data_parallel_shard_degree"; P="--parallelism.pipeline_parallel_degree"; PD="--parallelism.spmd_backend partial_dtensor"
B="--training.num-tokens-per-train-step 512 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
TITAN=/tmp/wt_pprt CFG=kimi_k3_debugmodel9 BATCH="$B" CELLS="dp1|1|$D 1 $PD
pp2|2|$D 1 $P 2 --parallelism.num-pp-microbatches 2 $PD" $MX pprt_fp32l9
O=$(ls -td /workspace/mx3_pprt_fp32l9_* | head -1)
python3 - "$O" <<'PY'
import re, sys
o = sys.argv[1]
def series(f):
    L, G = {}, {}
    for line in open(f, errors="ignore"):
        m = re.search(r"step: *(\d+)\s.*loss: *([0-9.]+).*grad_norm: *([0-9.]+)", line)
        if m: L[int(m.group(1))] = float(m.group(2)); G[int(m.group(1))] = float(m.group(3))
    return L, G
L1, G1 = series(f"{o}/dp1_measure.log"); L2, G2 = series(f"{o}/pp2_measure.log")
rows = [(s, L1[s], L2[s], abs(L2[s]-L1[s])/L1[s], G1[s], G2[s], abs(G2[s]-G1[s])/G1[s]) for s in sorted(L1) if s in L2 and s in G1 and s in G2]
for r in rows: print(f"step {r[0]}: loss {r[1]:.6f} vs {r[2]:.6f} (rel {r[3]:.2e}) | grad_norm {r[4]} vs {r[5]} (rel {r[6]:.2e})")
if rows: print(f"max relative loss diff {max(r[3] for r in rows):.2e}, max relative grad-norm diff {max(r[6] for r in rows):.2e}")
PY
grep -a -m1 "Error" $O/pp2_measure.log | grep -v lspci | cut -c1-160
echo "PP-FP32L9-DONE"
