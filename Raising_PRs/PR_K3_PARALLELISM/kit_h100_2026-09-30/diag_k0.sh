#!/bin/bash
# DEP-off step time diagnostics at seq 2048, M8, pp4 x vpp4, text only (DEPR_IMG_PER16=0): DEP off, K2.5 and the model
# without a tower, each warmed 2 steps on one cache and timed 20 steps on a copy. Results as diag_* in the DEP results dir.
M=~/mep; K=~/kit/overnight/dep_ratio; KD=~/kit/kit_h100_2026-09-29/dep; KH=~/kit/kit_h100_2026-09-30; V=$M/venv_src; W=$M/w/dep
export DEPR_LAYOUT=pp4vpp4; O=$M/results/dep_h100_$DEPR_LAYOUT; P=$O/progress.txt
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPR_RES=1024 DEPW_DIM=6144 DEPV_TOWER=k3 DEPR_AC=full NCCL_NVLS_ENABLE=0
export DEPR_SEQ=2048 DEPR_NMAX=1 DEPV_COST_RATIO=3.61 DEPR_IMG_PER16=0
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
run() {  # <name> <module> <config> <cache> <steps> [args...]
  local name=$1 mod=$2 cfg=$3 cache=$4 steps=$5; shift 5; local D=$O/$name; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$K:$KD:$KH:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    timeout 3000 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module $mod --config $cfg --training.steps $steps --metrics.log-freq 1 \
    --dump-folder $D/out "$@" > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out; note "$name rc=$rc"
}
B="--parallelism.num-pp-microbatches 8 --training.num-tokens-per-train-step 16384"
T0=$O/cache_diag; rm -rf $T0; mkdir -p $T0
CELLS="off:dep_ratio_local:w_dep_off k25:dep_ratio_local:w_dep_k25 notower:dep_diag_local:w_notower_off"
for x in $CELLS; do IFS=: read -r n mod cfg <<< "$x"; run diagwarm_k0_$n $mod $cfg $T0 2 $B; done
for x in $CELLS; do
  IFS=: read -r n mod cfg <<< "$x"; rm -rf $O/cc3; cp -r $T0 $O/cc3; run diag_k0_$n $mod $cfg $O/cc3 20 $B
done
rm -rf $O/cc3 $T0
