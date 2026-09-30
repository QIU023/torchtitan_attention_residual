#!/bin/bash
# The step-1 gap is dynamo recompiling the shared flex_attention with dynamic shapes on ranks that run both the tower and
# text full attention. Confirm: the numerics cells (r10 M8, seed 42, deterministic, one warm cache, 20 steps) with automatic
# dynamic shapes off, DEP off twice, K2.5 and bubble. Results in ~/mep/results/probe_static.
M=~/mep; K=~/kit/overnight/dep_ratio; KD=~/kit/kit_h100_2026-09-29/dep; KH=~/kit/kit_h100_2026-09-30; V=$M/venv_src; W=$M/w/dep
O=$M/results/probe_static; rm -rf $O; mkdir -p $O
export DEPR_LAYOUT=pp4vpp4 DEPR_DATA=/root/dep_data/t2i1024_k4 DEPR_RES=1024 DEPW_DIM=6144 DEPV_TOWER=k3 DEPR_AC=full \
  NCCL_NVLS_ENABLE=0 DEPR_SEQ=2048 DEPR_NMAX=1 DEPV_COST_RATIO=3.61 DEPR_IMG_PER16=7
B="--parallelism.num-pp-microbatches 8 --training.num-tokens-per-train-step 16384 --debug.seed 42 --debug.deterministic"
run() {  # <name> <config> <cache> <steps>
  local D=$O/$1; mkdir -p $D
  ( cd $W && PYTHONPATH=$K:$KD:$KH:. TORCHINDUCTOR_CACHE_DIR=$3/ic TRITON_CACHE_DIR=$3/tc \
    timeout 1500 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_static_local --config $2 --training.steps $4 --metrics.log-freq 1 \
    --dump-folder $D/out $B > $D/run.log 2>&1 )
  echo "$1 rc=$?" >> $O/summary.txt; rm -rf $D/out
}
C0=$O/cache0; mkdir -p $C0
for c in w_dep_off w_dep_k25 w_dep_bubble; do run warm_$c $c $C0 1; done
for x in off_a:w_dep_off off_b:w_dep_off k25:w_dep_k25 bubble:w_dep_bubble; do
  rm -rf $O/cc; cp -r $C0 $O/cc; run num_${x%%:*} ${x#*:} $O/cc 20
done
rm -rf $O/cc $C0; echo PROBE_DONE >> $O/summary.txt
