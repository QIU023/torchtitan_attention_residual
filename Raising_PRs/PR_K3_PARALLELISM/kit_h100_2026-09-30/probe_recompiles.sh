#!/bin/bash
# The step-1 gap starts at stage 2 on rank 2 (the first full-attention layer on a rank that encodes under DEP): log
# torch.compile recompiles in both modes (one step each, seed 42, deterministic, r10 M8). Results in ~/mep/results/probe_recompiles.
M=~/mep; K=~/kit/overnight/dep_ratio; KD=~/kit/kit_h100_2026-09-29/dep; KH=~/kit/kit_h100_2026-09-30; V=$M/venv_src; W=$M/w/dep
O=$M/results/probe_recompiles; rm -rf $O; mkdir -p $O
export DEPR_LAYOUT=pp4vpp4 DEPR_DATA=/root/dep_data/t2i1024_k4 DEPR_RES=1024 DEPW_DIM=6144 DEPV_TOWER=k3 DEPR_AC=full \
  NCCL_NVLS_ENABLE=0 DEPR_SEQ=2048 DEPR_NMAX=1 DEPV_COST_RATIO=3.61 DEPR_IMG_PER16=7 TORCH_LOGS=recompiles
C=$O/cache; mkdir -p $C
B="--parallelism.num-pp-microbatches 8 --training.num-tokens-per-train-step 16384 --debug.seed 42 --debug.deterministic"
for x in off:w_dep_off k25:w_dep_k25; do
  n=${x%%:*}; cfg=${x#*:}
  ( cd $W && PYTHONPATH=$K:$KD:$KH:. TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc \
    timeout 1500 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_ratio_local --config $cfg --training.steps 1 --metrics.log-freq 1 \
    --dump-folder $O/out_$n $B > $O/run_$n.log 2>&1 )
  echo "$n rc=$?" >> $O/summary.txt
  for r in 0 1 2 3; do
    echo "  rank$r recompiles: $(sed 's/\x1b\[[0-9;]*m//g' $O/run_$n.log | grep -a "^\[rank$r\]" | grep -a -c -i 'Recompiling function')" >> $O/summary.txt
  done
  sed 's/\x1b\[[0-9;]*m//g' $O/run_$n.log | grep -a -i -A3 'Recompiling function' | grep -a -E 'Recompiling|triggered by|guard' | cut -c1-220 | head -12 >> $O/summary.txt
  rm -rf $O/out_$n
done
rm -rf $C; echo PROBE_DONE >> $O/summary.txt
