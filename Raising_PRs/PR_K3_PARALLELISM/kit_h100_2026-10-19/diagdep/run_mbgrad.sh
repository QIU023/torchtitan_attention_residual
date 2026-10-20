#!/bin/bash
# Per-micro-batch replica tower gradients under DEP (K2.5 form, 1 step): new tree on venv_1006 (GPUs 0-3), old tree on
# venv_0928 (GPUs 4-7), each on a copy of its warmed cache.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; G=$S/diagdep
export DEPR_DATA=/workspace/dep_data/t2i1024_k4 DEPW_DIM=256 DEPV_TOWER=base DEPR_AC=full DEPR_LAYOUT=pp4vpp4 \
  DEPR_SEQ=2048 DEPR_RES=224 DEPR_NMAX=1 DEPN_MBS=16 DEPV_COST_RATIO=0.116 DEPN_DET=1 DEPN_STATIC=1 \
  PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
P=$G/progress_mb.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
one() {  # <tree> <venv> <pythonpath extra> <cache> <gpus>
  local tree=$1 venv=$2 extra=$3 C=$4 gpus=$5 O=$G/mb_$1; rm -rf $O $G/ccmb_$1; mkdir -p $O; cp -r $C $G/ccmb_$1
  ( cd $G/$tree && DEPN_STEPS=1 DUMP_DIR=$O/dump PYTHONPATH=$G/kit_$tree:$extra. TORCHINDUCTOR_CACHE_DIR=$G/ccmb_$1/ic \
      TRITON_CACHE_DIR=$G/ccmb_$1/tc CUDA_VISIBLE_DEVICES=$gpus timeout 2400 /workspace/$venv/bin/torchrun --nproc_per_node=4 \
      --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module dep_mbgrad \
      --config w_dep_k25 --output-dir $O/out > $O/run.log 2>&1 )
  local rc=$?; rm -rf $O/out $G/ccmb_$1
  note "$tree rc=$rc $(grep -a -o 'DEP_MBGRAD rank [0-9] micro-batches [^f]*' $O/run.log | tr '\n' ';')"
}
one new venv_1006 $S/main_pins_1016: $G/cache_new 0,1,2,3 &
one old venv_0928 "" $G/cache_old0928 4,5,6,7 &
wait
note MB_DONE
