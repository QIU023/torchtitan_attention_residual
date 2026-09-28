#!/bin/bash
# Sizing for PR A and DEP on the current debug model widened by dim / 256 (17 layers, blocks of 4),
# pp4 x vp2 on four 5060s, dims 512 / 768 / 1024, 4 steps each; per-rank peaks for the H100 extrapolation.
#  P: PR A's layout through the PP probe (AdamW, full AC, seq 2048, M16), on main (the heaviest tree).
#  D: the committed vit_dep cell (its own AC, optimizer and micro-batches), on the DEP head.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
R=$S/sizing2/res; mkdir -p $R; SHIM=$S/lbplan/pr5_torch_compat_shim.patch
M=$S/wt_s_main; D=/tmp/wt_depnew; V=/workspace/venv_bfx9/bin
for t in $M $D; do git -C $t apply $SHIM || { echo "shim failed on $t" >> $R/progress.txt; exit 1; }; done
p_cell() {  # <dim> <gpus>
  local name=p_d$1 O=$R/p_d$1; rm -rf $O; mkdir -p $O/cache
  ( cd $M && CUDA_VISIBLE_DEVICES=$2 PPMEM_OUT=$O/mem PPMEM_SEQ=2048 PPMEM_DIM=$1 PPMEM_STORE_TRACK=0 \
    PYTHONPATH=$S/sizing2:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
    timeout 2400 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module probe_lbw --config lb_probe --training.steps 4 --debug.seed 42 --debug.deterministic \
    --training.num-tokens-per-train-step 32768 --training.num-tokens-per-microbatch-per-dp-rank 2048 \
    --parallelism.pipeline-parallel-degree 4 --parallelism.pipeline-parallel-schedule Interleaved1F1B \
    --parallelism.num-pp-microbatches 16 --dump-folder $O/dump > $O/train.log 2>&1 ); local rc=$?
  echo "$(date +%H:%M:%S) $name rc=$rc" >> $R/progress.txt; rm -rf $O/cache $O/dump
}
d_cell() {  # <dim> <gpus>
  local name=d_d$1 O=$R/d_d$1; rm -rf $O; mkdir -p $O/cache
  ( cd $D && CUDA_VISIBLE_DEVICES=$2 DEPW_OUT=$O/mem DEPW_DIM=$1 \
    PYTHONPATH=$S/sizing2:/tmp/attn_gym_up:. TORCHINDUCTOR_CACHE_DIR=$O/cache/ic TRITON_CACHE_DIR=$O/cache/tc \
    timeout 2400 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module dep_widened --config depw_bubble_on --training.steps 4 --metrics.log_freq 1 \
    --debug.seed 42 --debug.deterministic --dump-folder $O/dump > $O/train.log 2>&1 ); local rc=$?
  echo "$(date +%H:%M:%S) $name rc=$rc" >> $R/progress.txt; rm -rf $O/cache $O/dump
}
p_cell 512 0,1,2,3 & d_cell 512 4,5,6,7 & wait
p_cell 768 0,1,2,3 & d_cell 768 4,5,6,7 & wait
p_cell 1024 0,1,2,3 & d_cell 1024 4,5,6,7 & wait
for t in $M $D; do git -C $t checkout -- torchtitan/distributed/pipeline_parallel.py torchtitan/distributed/utils.py; done
for t in $M $D; do echo "$t $(git -C $t status --short | wc -l)"; done > $R/worktrees_after.txt
echo done > $R/SIZING2_DONE
