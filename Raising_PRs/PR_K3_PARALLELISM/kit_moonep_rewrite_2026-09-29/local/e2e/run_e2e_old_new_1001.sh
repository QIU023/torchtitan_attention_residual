#!/bin/bash
# 10-01: is the MoonEP SAC fix numerically neutral? The h100 cell (fake MoonEP, the recipe's SelectiveAC) on the
# tree before the fix (dd69d2625, autograd Functions) and after it (A=24458aa6e, library ops), 10 steps each on copies
# of one warm cache, at once on GPUs 0-3 and 4-7, deterministic.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_moonep_rewrite_2026-09-29
F=$K/local/fake_moonep; E=${E:-$S/moonep_sac/e2e_old_new}; OLD=${OLD:-$S/wt_moonep_ab19}; NEW=${NEW:-$S/wt_moonep_0929n}
V=/workspace/venv_0928/bin; P=$E/progress.txt; mkdir -p $E
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
note "old $(git -C $OLD rev-parse --short HEAD) dirty=$(git -C $OLD status --short | wc -l) new $(git -C $NEW rev-parse --short HEAD) dirty=$(git -C $NEW status --short | wc -l)"
run() {  # <name> <tree> <cache> <steps> <gpus>
  local name=$1 tree=$2 cache=$3 steps=$4 gpus=$5; local D=$E/$name; rm -rf $D; mkdir -p $D
  ( cd $tree && PYTHONPATH=$F:$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    CUDA_VISIBLE_DEVICES=$gpus timeout 1200 $V/torchrun --nproc_per_node=4 --rdzv_backend c10d \
    --rdzv_endpoint=localhost:0 --role rank --tee 3 -m torchtitan.train --module moonep_local --config moonep_cell \
    --training.steps $steps --metrics.log_freq 1 --debug.seed 42 --debug.deterministic --dump-folder $D/out \
    > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/out; note "$name rc=$rc"
}
unset MOONEP_AC
C=$E/cache; rm -rf $C; mkdir -p $C
run warm_old $OLD $C 1 0,1,2,3; run warm_new $NEW $C 1 0,1,2,3
rm -rf $C.o $C.n; cp -r $C $C.o; cp -r $C $C.n
run old $OLD $C.o 10 0,1,2,3 & run new $NEW $C.n 10 4,5,6,7 & wait
rm -rf $C $C.o $C.n
note "old/new done"; echo done > $E/DONE
