#!/bin/bash
# The 16-rank 5D run (07-24 handoff sec 4 item 5): FSDP x TP x CP x PP all >1,
# plus an EP-folded second cell. Usage:
#   launch_5d.sh <NODE_RANK 0|1> <NCCL_IFACE> <MASTER_ADDR> [MASTER_PORT]
#
# Mesh order is [pp, dp_replicate, dp_shard, cp, tp] (parallel_dims.py:231):
# pp is OUTERMOST, so with default rank placement node 0 carries pp stage 0
# and node 1 stage 1 -- only the PP stage-boundary P2P crosses the LAN while
# dp_shard x cp x tp (= 8) stays intra-node. Do not reorder ranks.
#
# Gate (same standard as every matrix cell): descending finite loss AND
# rank-identical loss/grad_norm across all 16 ranks; step-1 compared against
# the <=8-rank projections in CP_TP_3D_VERIFICATION_2026-07-24.md.
set -eu
NODE_RANK=$1
IFACE=$2
MASTER_ADDR=${3:?pass node 0's address on the overlay interface}
MASTER_PORT=${4:-10193}
STEPS=${STEPS:-10}

cd /workspace/torchtitan_attention_residual/torchtitan
source /venv/main/bin/activate
export PYTHONPATH=$PWD

export NCCL_SOCKET_IFNAME=$IFACE
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

COMMON="--module kimi_k3 --config kimi_k3_debugmodel8h --checkpoint.no-enable \
 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 \
 --training.global-batch-size 4 --training.steps $STEPS \
 --parallelism.data_parallel_shard_degree 2 \
 --parallelism.tensor_parallel_degree 2 \
 --parallelism.context_parallel_degree 2 \
 --parallelism.pipeline_parallel_degree 2 \
 --parallelism.pipeline_parallel_schedule 1F1B"

run() {
  local name=$1; shift
  echo "=================== 5D CELL: $name (node_rank=$NODE_RANK) ==================="
  torchrun --nnodes 2 --nproc_per_node 8 --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
    -m torchtitan.train $COMMON --dump-folder "/workspace/out_5d/$name" "$@" 2>&1 \
  | sed -E 's/\x1b\[[0-9;]*m//g' \
  | grep -E "step: +[0-9]+ +loss|Traceback|RuntimeError|ValueError|AssertionError" \
  | grep -vE '\-4\.00000|\-2\.00000' || true
}

# cell 1: the pure 5D mesh (dp2 tp2 cp2 pp2 = 16)
run "5d_dp2tp2cp2pp2"

# cell 2: 5D + EP folded into dp_shard x cp (ep2) -- the full 6-axis story
run "5d_plus_ep2" --parallelism.expert_parallel_degree 2

echo "=================== 5D DONE (node_rank=$NODE_RANK) ==================="
