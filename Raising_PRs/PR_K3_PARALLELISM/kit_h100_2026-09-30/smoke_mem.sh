#!/bin/bash
# Memory check before the DEP campaign: the heaviest level (seq SEQ, cap CAP) with FullAC, M MBS, 2 steps of DEP off and
# of bubble in the layout DEPR_LAYOUT (default pp4vpp4); prints rc, the step lines and every rank's reserved memory.
M=~/mep; K=~/kit/overnight/dep_ratio; KD=~/kit/kit_h100_2026-09-29/dep; V=$M/venv_src; W=$M/w/dep
export DEPR_LAYOUT=${DEPR_LAYOUT:-pp4vpp4}; O=$M/results/dep_smoke_$DEPR_LAYOUT; mkdir -p $O
export DEPR_DATA=/root/dep_data/t2i1024_k4 DEPR_RES=1024 DEPW_DIM=6144 DEPV_TOWER=k3 DEPR_AC=full NCCL_NVLS_ENABLE=0
export DEPR_SEQ=${SEQ:-8192} DEPR_NMAX=${CAP:-1} DEPV_COST_RATIO=${RATIO:-0.44}
MB=${MBS:-4}
for c in w_dep_off w_dep_bubble; do
  D=$O/${c}_s${DEPR_SEQ}_c${DEPR_NMAX}_m$MB; rm -rf $D; mkdir -p $D
  ( cd $W && PYTHONPATH=$K:$KD:. TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
    timeout 1500 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_ratio_local --config $c --training.steps 2 --metrics.log-freq 1 \
    --parallelism.num-pp-microbatches $MB --training.num-tokens-per-train-step $((MB * DEPR_SEQ)) \
    --dump-folder $D/out > $D/run.log 2>&1 )
  echo "$c $DEPR_LAYOUT seq $DEPR_SEQ cap $DEPR_NMAX M$MB rc=$?"
  sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o -E '\[rank[0-9]\].*step: +2 .*memory: +[0-9.]+GiB\([0-9.]+%\)' | sed -E 's/.*\[rank([0-9])\].*memory: +([0-9.]+GiB\([0-9.]+%\)).*/rank\1 \2/' | sort -u
  sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -m2 -E 'OutOfMemory|Error:' | cut -c1-200
  sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -m1 -o 'vision_dep: .*'
  rm -rf $D/out
done
echo SMOKE_DONE
