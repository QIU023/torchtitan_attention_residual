#!/bin/bash
# PR A on 4 x H100 (09-30 box): the rebased #4656 (pp_review_optimize e66a9442b) against PR A (345e9e00a, the minimal
# version), the Kimi K3 debug model widened to dim PPMEM_DIM (6144), 16 micro-batches x 2048 tokens of c4 text, FullAC,
# AdamW, seed 42, deterministic, STEPS (100) steps, the per-action block account in step 5. Per layout: one warm cache (a
# step of each tree), then each tree on its own copy, one after the other; pp4vp2 also traces one step of each (step 10).
# Usage: run_pra_h100.sh <layout> [<layout> ...]; results in ~/mep/results/pra_h100.
M=~/mep; K=~/kit/kit_h100_2026-09-30/pra; KT=~/kit/kit_h100_2026-09-29/pra; V=$M/venv_src
A=$M/w/pra_base; B=$M/w/pra; O=${OUT:-$M/results/pra_h100}; P=$O/progress.txt; mkdir -p $O
DIM=${PPMEM_DIM:-6144}; STEPS=${STEPS:-100}; export NCCL_NVLS_ENABLE=0
# titan's run_train.sh default; without it both trees OOM at dim 6144 on about 21 GiB reserved but unallocated.
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
declare -A PP=([pp4vp2]=4 [pp4vp4]=4 [pp2vp2]=2 [pp2vp4]=2 [dp2pp2vp2]=2)
declare -A DP=([pp4vp2]=1 [pp4vp4]=1 [pp2vp2]=1 [pp2vp4]=1 [dp2pp2vp2]=2)
declare -A ST=([pp4vp2]= [pp4vp4]=16 [pp2vp2]= [pp2vp4]=8 [dp2pp2vp2]=)
run() {  # <name> <tree> <cache> <steps> <layout> <trace step> [extra args...]
  local name=$1 tree=$2 cache=$3 steps=$4 L=$5 trace=$6; shift 6; local D=$O/$name; rm -rf $D; mkdir -p $D
  local pp=${PP[$L]} dp=${DP[$L]}; local n=$((pp * dp)); local gpus=$(seq -s, 0 $((n - 1)))
  ( cd $tree && CUDA_VISIBLE_DEVICES=$gpus PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    PPMEM_OUT=$D/mem PPMEM_DIM=$DIM PPMEM_SEQ=2048 PPMEM_STAGES=${ST[$L]} PPMEM_ACTION_TRACE=$trace \
    timeout 3000 $V/bin/torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module probe_bound --config lb_probe --training.steps $steps --debug.seed 42 --debug.deterministic \
    --metrics.log-freq 1 --training.num-tokens-per-train-step $((16 * 2048 * dp)) --training.num-tokens-per-microbatch-per-dp-rank 2048 \
    --parallelism.pipeline-parallel-degree $pp --parallelism.pipeline-parallel-schedule Interleaved1F1B \
    --parallelism.num-pp-microbatches 16 --parallelism.data-parallel-shard-degree $dp --dump-folder $D/dump "$@" > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/dump/checkpoint
  note "$name rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -m1 -o 'OutOfMemoryError.\{0,80\}')"
}
note "4656 $(git -C $A rev-parse --short HEAD) pra $(git -C $B rev-parse --short HEAD) dim $DIM steps $STEPS torch $($V/bin/python -c 'import torch; print(torch.__version__)')"
for L in "$@"; do
  C=$O/cache_$L; rm -rf $C; mkdir -p $C
  run warm4656_$L $A $C 1 $L 0; run warmpra_$L $B $C 1 $L 0
  for x in 4656:$A pra:$B; do
    rm -rf $O/cc; cp -r $C $O/cc; run ${x%%:*}_$L ${x#*:} $O/cc $STEPS $L 5; rm -rf $O/${x%%:*}_$L/dump
  done
  if [ "$L" = pp4vp2 ]; then
    for x in 4656:$A pra:$B; do
      rm -rf $O/cc; cp -r $C $O/cc
      run trace${x%%:*}_$L ${x#*:} $O/cc 12 $L 0 --profiler.enable-profiling --profiler.profile-freq 10 \
        --profiler.profiler-warmup 2 --profiler.profiler-active 1 --profiler.save-traces-folder traces
      t=$(find $O/trace${x%%:*}_$L/dump -name "rank0_trace.json*" | head -1)
      [ -n "$t" ] && $V/bin/python $KT/analyze_trace.py $(dirname $t) > $O/trace${x%%:*}_$L/analysis.txt 2>&1
    done
  fi
  rm -rf $O/cc $C
  $V/bin/python $K/tab_bound.py $O $L > $O/tab_$L.md 2>&1
done
note "pra campaign done: $*"; echo done > $O/DONE
