#!/bin/bash
# Frozen-router row on the new base: lr 0, dp1, 15 steps, sign-step vs QB, load probes on; GPU 7 only.
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export VENV=/workspace/venv_bfx9 PYPRE=/tmp/attn_gym_up MEASURE_STEPS=15 WARM_STEPS=1 SEED_ROOT=/workspace/.mx3_seeds_qb65 SEED_CFG=kimi_k3_debugmodel
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
CUDA_VISIBLE_DEVICES=7 TRITON_CACHE_DIR=/workspace/.triton_qb100_frozen TITAN=/tmp/wt_qbrun4 CFG=kimi_k3_debugmodel_probe_frozen BATCH="$B" CELLS="dp1_frozen_ctrl|1|--parallelism.data_parallel_shard_degree 1 --parallelism.spmd_backend partial_dtensor" $MX qb100_frozen_ctrl
CUDA_VISIBLE_DEVICES=7 TRITON_CACHE_DIR=/workspace/.triton_qb100_frozen TITAN=/tmp/wt_qbrun4 CFG=kimi_k3_debugmodel_qb_probe_frozen BATCH="$B" CELLS="dp1_frozen_qb|1|--parallelism.data_parallel_shard_degree 1 --parallelism.spmd_backend partial_dtensor" $MX qb100_frozen_qb
python3 - <<'PY'
import re, glob, statistics
for d in sorted(glob.glob("/workspace/mx3_qb100_frozen_*")):
    P = {}
    for f in glob.glob(d + "/*_measure.log"):
        for line in open(f, errors="ignore"):
            m = re.search(r"LOADPROBE step=(\d+) layer=(\d+) tokens=(\d+) cv=([0-9.]+)", line)
            if m: P.setdefault(int(m.group(1)), []).append(float(m.group(4)))
    cv = lambda s: statistics.mean(P[s]) if s in P else float("nan")
    print(f"{d.split('/')[-1]}: cv step1={cv(1):.2f} step5={cv(5):.2f} step10={cv(10):.2f} step15={cv(15):.2f}")
PY
echo "QB-FROZEN-DONE"
