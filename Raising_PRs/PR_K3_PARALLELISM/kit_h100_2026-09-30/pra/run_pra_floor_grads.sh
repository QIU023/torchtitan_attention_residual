#!/bin/bash
# 10-01 PR A additions for body v4 on the 09-30 H100, after the MoonEP campaign: the step-1 gradients of #4656
# (without this PR) and PR A at pp4 x vpp2, each on a copy of one warm cache, hashed per parameter before clipping
# (PPMEM_GRAD_DUMP), then the noise-floor row: #4656 on a fresh, empty compile cache, 100 steps. Everything else is
# run_pra_h100.sh's pp4vp2 cell (dim 6144, 16 x 2048 c4 tokens, Interleaved1F1B, FullAC, AdamW, seed 42, deterministic).
M=~/mep; K=~/kit/kit_h100_2026-09-30/pra; V=$M/venv_src
A=$M/w/pra_base; B=$M/w/pra; O=${OUT:-$M/results/pra_floor_grads}; P=$O/progress.txt; mkdir -p $O
DIM=6144; export NCCL_NVLS_ENABLE=0
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
WAIT=${WAIT:-$M/results/moonep_sac/DONE}
note "waiting for $WAIT"
until [ -f $WAIT ]; do sleep 30; done
note "4656 $(git -C $A rev-parse --short HEAD) dirty=$(git -C $A status --short | wc -l) pra $(git -C $B rev-parse --short HEAD) dirty=$(git -C $B status --short | wc -l) dim $DIM torch $($V/bin/python -c 'import torch; print(torch.__version__)')"
run() {  # <name> <tree> <cache> <steps> <trace step> [NAME=VALUE ...]
  local name=$1 tree=$2 cache=$3 steps=$4 trace=$5; shift 5; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $tree && env "$@" CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic \
    TRITON_CACHE_DIR=$cache/tc PPMEM_OUT=$D/mem PPMEM_DIM=$DIM PPMEM_SEQ=2048 PPMEM_STAGES= PPMEM_ACTION_TRACE=$trace \
    timeout 3000 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module probe_bound --config lb_probe --training.steps $steps --debug.seed 42 \
    --debug.deterministic --metrics.log-freq 1 --training.num-tokens-per-train-step $((16 * 2048)) \
    --training.num-tokens-per-microbatch-per-dp-rank 2048 --parallelism.pipeline-parallel-degree 4 \
    --parallelism.pipeline-parallel-schedule Interleaved1F1B --parallelism.num-pp-microbatches 16 \
    --parallelism.data-parallel-shard-degree 1 --dump-folder $D/dump > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/dump; note "$name rc=$rc"
}
C=$O/cache_warm; rm -rf $C; mkdir -p $C
run warm4656 $A $C 1 0; run warmpra $B $C 1 0
rm -rf $C.a $C.b; cp -r $C $C.a; cp -r $C $C.b
run grads4656 $A $C.a 1 0 PPMEM_GRAD_DUMP=$O/grads4656/hashes
run gradspra $B $C.b 1 0 PPMEM_GRAD_DUMP=$O/gradspra/hashes
rm -rf $C $C.a $C.b
$V/bin/python $K/cmp_grads.py $O/grads4656/hashes $O/gradspra/hashes > $O/grads_cmp.txt 2>&1
note "grads: $(head -1 $O/grads_cmp.txt)"
F=$O/cache_fresh; rm -rf $F; mkdir -p $F
run floor4656_pp4vp2 $A $F 100 5
rm -rf $F
note "pra floor/grads done"; echo done > $O/DONE
