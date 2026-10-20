#!/bin/bash
# #4380 diagnosis on the 5060: main e61df55b6 against the PR a2a97b88a with dynamic CP on and no image split, CP=2, dp1,
# typechecking on, seed 42, deterministic, one seed checkpoint; 2 steps per cell with clipping on/off and the recipe
# optimizer (DistMuon + AdamW) or AdamW alone, each on a copy of one cache warmed by 2 steps of main and of the PR.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; D=$S/diag4380; K=$D/kit
source /workspace/venv_1006/bin/activate
P=$D/progress.txt; note() { echo "$(date +%H:%M:%S) $*" >> $P; }
run() {  # <name> <tree> <gpus> <cache> <steps> <envs>
  local name=$1 W=$2 gpus=$3 C=$4 steps=$5 envs=$6 O=$D/$1; rm -rf $O; mkdir -p $O
  ( cd $W && env $(echo $envs | tr ',' ' ') CP_STEPS=$steps CP_SEED=$D/seed/step-0 DUMP_DIR=$O/dump \
      PYTHONPATH=$K:$S/main_pins_1016:. TORCHINDUCTOR_CACHE_DIR=$C/ic TRITON_CACHE_DIR=$C/tc \
      CUDA_VISIBLE_DEVICES=$gpus NGPU=2 LOG_RANK=0 MODULE=upd1_probe CONFIG=cell \
      timeout 1800 ./run_train.sh --output-dir $O/out > $O/run.log 2>&1 )
  local rc=$?; rm -rf $O/out
  note "$name rc=$rc seed_load=$(grep -a -c 'Loading the checkpoint' $O/run.log) patched=$(grep -a -c 'upd1_probe\] rank' $O/run.log) $(grep -a -o 'UPD1_GRAD_NORM [^ ]*' $O/run.log) $(sed 's/\x1b\[[0-9;]*m//g' $O/run.log | grep -a -o 'step: *[0-9]* *loss: *[0-9.]*' | tr '\n' ' ')"
}
note "start main $(ls $D/main/torchtitan | wc -l) pr $(ls $D/pr/torchtitan/models/kimi_k3/vision_cp 2>/dev/null | wc -l) vision_cp files"
if [ ! -d $D/seed/step-0 ]; then
  ( cd $D/main && CP_SEED_OUT=$D/seed PYTHONPATH=$K:$S/main_pins_1016:. CUDA_VISIBLE_DEVICES=0 NGPU=1 LOG_RANK=0 \
      MODULE=upd1_probe CONFIG=seed timeout 1800 ./run_train.sh --output-dir $D/seed_run > $D/seed.log 2>&1 )
  note "seed rc=$? $(ls $D/seed 2>&1 | tr '\n' ' ')"; rm -rf $D/seed_run
fi
[ -d $D/seed/step-0 ] || { note "no seed, stop"; exit 1; }
W=$D/cache_w; mkdir -p $W/ic $W/tc
run warm_main $D/main 0,1 $W 2 CP_MODE=allgather
run warm_pr $D/pr 0,1 $W 2 CP_MODE=allgather,CP_MINP=1000000000
rm -rf $D/warm_main/dump $D/warm_pr/dump
MAIN=CP_MODE=allgather; WHOLE=CP_MODE=allgather,CP_MINP=1000000000; OFF=CP_MODE=allgather,CP_MINP=off
lane() {  # <gpus> <name:tree:envs>...
  local gpus=$1; shift
  for spec in "$@"; do IFS=: read -r name tree envs <<< "$spec"
    rm -rf $D/cc_$name; cp -r $W $D/cc_$name; run $name $D/$tree $gpus $D/cc_$name 2 $envs; rm -rf $D/cc_$name
  done
}
lane 0,1 main__clipon_recipe:main:$MAIN prwhole__clipon_recipe:pr:$WHOLE proff__clipon_recipe:pr:$OFF &
lane 2,3 main__clipoff_recipe:main:$MAIN,CLIP_OFF=1 prwhole__clipoff_recipe:pr:$WHOLE,CLIP_OFF=1 &
lane 4,5 main__clipon_adamw:main:$MAIN,CP_OPT=adamw prwhole__clipon_adamw:pr:$WHOLE,CP_OPT=adamw &
lane 6,7 main__clipoff_adamw:main:$MAIN,CP_OPT=adamw,CLIP_OFF=1 prwhole__clipoff_adamw:pr:$WHOLE,CP_OPT=adamw,CLIP_OFF=1 &
wait
python $K/analyze_upd1.py $D > $D/analysis.txt 2>&1
note "DIAG_DONE analysis rc=$?"
