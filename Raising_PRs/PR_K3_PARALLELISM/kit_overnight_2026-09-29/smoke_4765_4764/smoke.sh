#!/bin/bash
# T1f smoke on 4 x 5060, torch 2.15.0.dev20260928: the restacked #4765 019462171 and #4764 c4afb61f4 on pp4 x vpp2,
# the debug model widened to dim 2048, 16 micro-batches x 2048 tokens, FullAC, 6 steps, one cache; rc and every rank's peak.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=$(cd "$(dirname "$0")" && pwd); V=/workspace/venv_0928; R=$S/smoke_4765; P=$R/progress.txt; mkdir -p $R
A=$S/wt_4765_v3; B=$S/wt_restack; C=$R/cache; mkdir -p $C
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "4765 $(git -C $A rev-parse --short HEAD) 4764 $(git -C $B rev-parse --short HEAD)"
cell() {  # <name> <tree> [VAR=val ...]
  local name=$1 tree=$2; shift 2; local D=$R/$name; rm -rf $D; mkdir -p $D
  ( cd $tree && env "$@" PPMEM_OUT=$D/mem PPMEM_DIM=2048 PPMEM_SEQ=2048 PYTHONPATH=$K:. CUDA_VISIBLE_DEVICES=0,1,2,3 \
    TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc timeout 2400 $V/bin/torchrun --nproc_per_node=4 \
    --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module probe_lbw --config lb_probe \
    --training.steps 6 --debug.seed 42 --debug.deterministic --metrics.log-freq 1 \
    --training.num-tokens-per-train-step 32768 --training.num-tokens-per-microbatch-per-dp-rank 2048 \
    --parallelism.pipeline-parallel-degree 4 --parallelism.pipeline-parallel-schedule Interleaved1F1B \
    --parallelism.num-pp-microbatches 16 --parallelism.data-parallel-shard-degree 1 --dump-folder $D/dump > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/dump; note "$name rc=$rc"
}
cell none_4764 $B PPMEM_STORE_TRACK=0
cell o4765_all $A PPMEM_STORE_TRACK=0 PPMEM_CPU_OFFLOAD=all
cell b4764_planned $B PPMEM_STORE_TRACK=0 PPMEM_CPU_OFFLOAD=planned
cell b4764_balance_tcp $B PPMEM_STORE_TRACK=0 PPMEM_BALANCE=1 PPMEM_REMOTE_PROTOCOL=tcp
cell b4764_planned_balance_tcp $B PPMEM_STORE_TRACK=0 PPMEM_CPU_OFFLOAD=planned PPMEM_BALANCE=1 PPMEM_REMOTE_PROTOCOL=tcp
rm -rf $C
note "smoke done"
