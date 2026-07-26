#!/bin/bash
# 2-node NCCL smoke launcher. Usage: launch_smoke.sh <NODE_RANK 0|1> <NCCL_IFACE> [MASTER_ADDR] [MASTER_PORT]
#
# MASTER_ADDR defaults to node 0's OVERLAY address -- pass it explicitly once
# the overlay is up (find it with `ip -4 addr show <NCCL_IFACE>` on node 0).
# The rendezvous port must be reachable from node 1: on node 0 use a
# direct-forwarded container port (10193 is the freed syncthing port).
set -eu
NODE_RANK=$1
IFACE=$2
MASTER_ADDR=${3:?pass node 0's address on the overlay interface}
MASTER_PORT=${4:-10193}

cd /workspace/torchtitan_attention_residual/torchtitan
source /venv/main/bin/activate
export PYTHONPATH=$PWD

export NCCL_SOCKET_IFNAME=$IFACE   # the overlay iface -- NOT the docker bridge
export NCCL_IB_DISABLE=1           # no IB on this fabric; don't let NCCL probe
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}   # set INFO for the first run

torchrun \
  --nnodes 2 --nproc_per_node 8 --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" --master_port "$MASTER_PORT" \
  ../phase13_k3like_48b_posttrain/multinode/nccl_smoke_2node.py
