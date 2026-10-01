#!/bin/bash
# 10-01 night, DEP on 8 x 5060, phase 2a: the planner's placement at each cost ratio. pp4 x vpp4, dim 1024 text with the
# non-widened tower, full AC, M16, bubble mode for 2 steps per level (the plan is made from the cost ratio and the
# schedule, so its log line is all this needs). LEVELS="name:seq:cap:ratio:mbs:res ..." as in phase2_dep.sh; levels of
# one seq share a compile cache.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_overnight_2026-09-29/dep_ratio
KD=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_h100_2026-09-29/dep
V=/workspace/venv_0928; W=${W:-$S/wt_dep_1001}; O=${OUT:-$S/dep_1001/phase2a}; P=$S/dep_1001/progress.txt; mkdir -p $O
export DEPR_LAYOUT=pp4vpp4 DEPR_DATA=/workspace/dep_data/t2i1024_k4 DEPW_DIM=${DEPW_DIM:-1024}
export DEPV_TOWER=${DEPV_TOWER:-base} DEPR_AC=full
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "phase2a dep $(git -C $W rev-parse --short HEAD) dim $DEPW_DIM tower $DEPV_TOWER levels: $LEVELS"
for lvl in $LEVELS; do
  IFS=: read -r name seq cap ratio mbs res <<< "$lvl"
  export DEPR_SEQ=$seq DEPR_NMAX=$cap DEPV_COST_RATIO=$ratio DEPR_RES=$res
  D=$O/place_$name; C=$O/cache_s$seq; rm -rf $D; mkdir -p $D $C
  ( cd $W && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$K:$KD:. TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc \
    DEPN_MEM_OUT=$D/mem timeout 1800 $V/bin/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 \
    --role rank --tee 3 -m torchtitan.train --module dep_ratio_local --config w_dep_bubble --training.steps 2 \
    --metrics.log-freq 1 --parallelism.num-pp-microbatches $mbs --training.num-tokens-per-train-step $((mbs * seq)) \
    --dump-folder $D/out > $D/run.log 2>&1 )
  rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out
  note "place $name (seq $seq, ${res} px, cap $cap, ratio $ratio) rc=$rc $(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'vision_dep: .*' | head -1)"
done
note "phase2a done"; echo done > $O/DONE
