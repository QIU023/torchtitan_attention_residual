#!/bin/bash
# PR 4656 list-only (7dea4cbaf) against main (838e6962e) on a 4 x H100 box (torch 68e0ae4, the 09-28 nightly source).
#   memory: dp 1, 262144 tokens per step in 65536-token micro-batches, AC none / selective / full / region, 10 steps,
#     one AC mode per GPU in parallel;
#   numerics: 100 steps, t1 (dp 1, 256 tokens, main twice for the noise floor), tp2 x ep2 with SP on and off,
#     dp2 x tp2 x ep2 (512 tokens), and the B200 suite's mm cell with type checking off; one cell at a time.
# Every pair runs on copies of one cache warmed by a 1-step run of main and of list in that configuration.
# Layout: venv ~/mep/venv_src, fork clone ~/mep/tt, this kit in ~/kit/kit_4656_list_2026-10-04.
M=~/mep; V=$M/venv_src; K=~/kit/kit_4656_list_2026-10-04
O=$M/results/m4656_list; mkdir -p $O; P=$O/progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
git -C $M/tt fetch -q upstream main; git -C $M/tt fetch -q origin attnres_review1 k3_attnres_ac_reuse
for x in main:838e6962e list:7dea4cbaf; do W=$M/w/ar_${x%%:*}; [ -d $W ] || git -C $M/tt worktree add -q --detach $W ${x#*:}
  note "${x%%:*} $(git -C $W rev-parse --short HEAD) dirty=$(git -C $W status --short | wc -l)"; done
note "torch $(python -c 'import torch;print(torch.__version__)') gpus $(nvidia-smi --query-gpu=name --format=csv,noheader | sort | uniq -c | xargs)"
run() {  # <tree> <name> <gpus> <cache> <env...>
  local T=$M/w/ar_$1 name=$2 gpus=$3 cache=$4; shift 4; local D=$O/$name; rm -rf $D; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l)
  ( cd $T && env "$@" PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc CUDA_VISIBLE_DEVICES=$gpus \
    NGPU=$n LOG_RANK=$(seq -s, 0 $((n - 1))) MODULE=attnres_h100_1004 timeout 3600 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out; note "$name rc=$rc last step=$(sed 's/\x1b\[[0-9;]*m//g' $D/run.log | grep -a -o 'step: *[0-9]*' | awk '{print $2}' | sort -n | tail -1)"
}
pair() {  # <name> <gpus> <extra measured runs: "main2" or ""> <env...>
  local name=$1 gpus=$2 extra=$3; shift 3; local C=$O/cache_$name; rm -rf $C; mkdir -p $C
  run main warm_main_$name $gpus $C "$@" AR_STEPS=1
  run list warm_list_$name $gpus $C "$@" AR_STEPS=1
  for t in main list $extra; do rm -rf $C.c; cp -r $C $C.c; run ${t%2} ${t}_$name $gpus $C.c "$@"; done
  rm -rf $C $C.c
}
g=0
for ac in none selective full region; do
  pair mem_$ac $g "" CONFIG=cell AR_AC=$ac AR_TOKENS_STEP=262144 AR_TOKENS_MB=65536 AR_STEPS=10 &
  g=$((g + 1))
done
wait
pair t1 0 main2 CONFIG=cell AR_STEPS=100
pair t1_tp2ep2_sp 0,1 "" CONFIG=cell AR_TP=2 AR_EP=2 AR_SP=1 AR_STEPS=100
pair t1_tp2ep2_nosp 0,1 "" CONFIG=cell AR_TP=2 AR_EP=2 AR_SP=0 AR_STEPS=100
pair t2_dp2tp2ep2_sp 0,1,2,3 "" CONFIG=cell AR_DP=2 AR_TP=2 AR_EP=2 AR_SP=1 AR_TOKENS_STEP=512 AR_STEPS=100
pair mm 0,1,2,3 "" CONFIG=mm_cell AR_STEPS=100
note "all done"; echo done > $O/DONE
