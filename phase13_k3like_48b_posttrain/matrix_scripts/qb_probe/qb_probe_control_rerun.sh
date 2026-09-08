#!/bin/bash
# Old base: rerun the CONTROL arm with core's sign rule active, on the pair's existing cache (one cache with the QB arm already run).
PAIR=$1 NPROC=$2 GPUS=$3; shift 3
MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh
export SEED_ROOT=/workspace/.mx3_seeds_main SEED_CFG=kimi_k3_debugmodel PYPRE=/tmp/attn_gym_up MEASURE_STEPS=30
export CUDA_VISIBLE_DEVICES=$GPUS TRITON_CACHE_DIR=/workspace/.triton_qbprobe_$PAIR
B="--training.num-tokens-per-train-step 8192 --training.num-tokens-per-microbatch-per-dp-rank 256"
CELLS="$PAIR|$NPROC|$* --parallelism.spmd_backend partial_dtensor"
OLD=$(ls -td /workspace/mx3_qbprobe_${PAIR}_control_* | head -1)
rm -rf /workspace/qb_probe_routes/${PAIR}_control
QB_PROBE_ROUTE_DIR=/workspace/qb_probe_routes/${PAIR}_control INDUCTOR_SEED_CACHE=$OLD/inductor TITAN=/tmp/wt_qbrebase CFG=kimi_k3_debugmodel_probe BATCH="$B" CELLS="$CELLS" $MX qbprobe2_${PAIR}_control
echo "CONTROL-RERUN-DONE $PAIR $(ls -td /workspace/mx3_qbprobe2_${PAIR}_control_* | head -1)"
