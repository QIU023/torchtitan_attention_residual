#!/bin/bash
# The QB evidence run under the matrix's real conditions: mx3 (seed checkpoint, warm+measure passes), the
# partial_dtensor backend the 09-07 table used, one inductor cache per pair, 30 measured steps, load probes on.
# args: pair nproc gpus cellflags...
PAIR=$1 NPROC=$2 GPUS=$3; shift 3
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export VENV=/workspace/venv_bfx9 SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up MEASURE_STEPS=30
export CUDA_VISIBLE_DEVICES=$GPUS TRITON_CACHE_DIR=/workspace/.triton_qbtip_$PAIR
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
CELLS="$PAIR|$NPROC|$* --parallelism.spmd_backend partial_dtensor"
QB_PROBE_ROUTE_DIR=/workspace/qb_probe_routes_tip/${PAIR}_control TITAN=/tmp/wt_qbtip CFG=kimi_k3_debugmodel_probe BATCH="$B" CELLS="$CELLS" $MX qbtip_${PAIR}_control
CTRL=$(ls -td /workspace/mx3_qbtip_${PAIR}_control_* | head -1)
QB_PROBE_ROUTE_DIR=/workspace/qb_probe_routes_tip/${PAIR}_qb INDUCTOR_SEED_CACHE=$CTRL/inductor TITAN=/tmp/wt_qbtip CFG=kimi_k3_debugmodel_qb_probe BATCH="$B" CELLS="$CELLS" $MX qbtip_${PAIR}_qb
echo "PAIR-DONE $PAIR $CTRL $(ls -td /workspace/mx3_qbtip_${PAIR}_qb_* | head -1)"
