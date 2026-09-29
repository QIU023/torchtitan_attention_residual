#!/bin/bash
# T1d on 8 x 5060, torch 2.15.0.dev20260928+cu130: #4656 5d469fdf3 against the rebuilt PR A 4ddf8e912, the debug model
# widened to PPMEM_DIM, 16 micro-batches x 2048 tokens of c4 text, FullAC, AdamW, seed 42, deterministic, 20 steps,
# the per-action block account in step 5. Per layout: one warm cache (a step of each tree), then both trees at once
# on disjoint GPUs, each on its own copy. Usage: run_bound.sh <layout> [<layout> ...]
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=$(cd "$(dirname "$0")" && pwd); V=/workspace/venv_0928; R=${R:-$S/bound}; P=$R/progress.txt; mkdir -p $R
A=${A:-$S/wt_4656}; B=${B:-$S/wt_pra_v3}; DIM=${PPMEM_DIM:-2048}; STEPS=${STEPS:-20}; NA=${NA:-4656}; NB=${NB:-pra}
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
declare -A PP=([pp4vp2]=4 [pp4vp4]=4 [pp2vp2]=2 [pp2vp4]=2 [dp2pp2vp2]=2)
declare -A DP=([pp4vp2]=1 [pp4vp4]=1 [pp2vp2]=1 [pp2vp4]=1 [dp2pp2vp2]=2)
declare -A ST=([pp4vp2]= [pp4vp4]=16 [pp2vp2]= [pp2vp4]=8 [dp2pp2vp2]=)
run() {  # <name> <tree> <gpus> <cache> <steps> <layout> <trace step>
  local name=$1 tree=$2 gpus=$3 cache=$4 steps=$5 L=$6 trace=$7; local D=$R/$name; rm -rf $D; mkdir -p $D
  local n=$(echo $gpus | tr ',' '\n' | wc -l) pp=${PP[$L]} dp=${DP[$L]}
  ( cd $tree && CUDA_VISIBLE_DEVICES=$gpus PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    PPMEM_OUT=$D/mem PPMEM_DIM=$DIM PPMEM_SEQ=2048 PPMEM_STAGES=${ST[$L]} PPMEM_ACTION_TRACE=$trace \
    timeout 2400 $V/bin/torchrun --nproc_per_node=$n --rdzv_backend c10d --rdzv_endpoint=localhost:0 --role rank --tee 3 \
    -m torchtitan.train --module probe_bound --config lb_probe --training.steps $steps --debug.seed 42 --debug.deterministic \
    --metrics.log-freq 1 --training.num-tokens-per-train-step $((16 * 2048 * dp)) --training.num-tokens-per-microbatch-per-dp-rank 2048 \
    --parallelism.pipeline-parallel-degree $pp --parallelism.pipeline-parallel-schedule Interleaved1F1B \
    --parallelism.num-pp-microbatches 16 --parallelism.data-parallel-shard-degree $dp --dump-folder $D/dump > $D/run.log 2>&1 )
  local rc=$?; echo "rc=$rc" > $D/rc; rm -rf $D/dump; note "$name rc=$rc"
}
note "$NA $(git -C $A rev-parse --short HEAD) $NB $(git -C $B rev-parse --short HEAD) dirty $(git -C $B status --short | wc -l) dim $DIM steps $STEPS torch $($V/bin/python -c 'import torch; print(torch.__version__)')"
for L in "$@"; do
  n=$((${PP[$L]} * ${DP[$L]})); g1=$(seq -s, 0 $((n - 1))); g2=$(seq -s, 4 $((4 + n - 1)))
  C=$R/cache_$L; rm -rf $C; mkdir -p $C
  run warm${NA}_$L $A $g1 $C 1 $L 0; run warm${NB}_$L $B $g1 $C 1 $L 0
  rm -rf $C.a $C.b; cp -r $C $C.a; cp -r $C $C.b
  run ${NA}_$L $A $g1 $C.a $STEPS $L 5 & run ${NB}_$L $B $g2 $C.b $STEPS $L 5 & wait
  rm -rf $C $C.a $C.b
done
note "bound campaign done: $*"
