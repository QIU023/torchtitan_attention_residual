#!/bin/bash
# 10-05, after h100_moonep_split_1005.sh: the numbers for the body on the PR head to be, moonep_review1 = f210edd3a.
#   1. microbench, old head 5e4596dc7 and f210edd3a in one session: the expert op (uniform and skewed routing) and the
#      whole MoE layer;
#   2. end-to-end, Kimi K3 debug model, seq 512, 20 steps, deterministic, one cache warmed by every cell: standard EP
#      (twice), MoonEP, MoonEP under FullAC at FSDP 4 x EP 4, and standard EP and MoonEP at dp_shard 4 x EP 2 and HSDP.
M=~/mep; V=$M/venv_src; K=~/kit/kit_moonep_perf_2026-10-03
O=$M/results/moonep_head; mkdir -p $O; P=$O/progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
until [ -f $M/results/moonep_split/DONE ]; do sleep 20; done
. $V/bin/activate
for t in mep_old mep_c5s; do note "$t $(git -C $M/w/$t rev-parse --short HEAD) dirty=$(git -C $M/w/$t status --short | wc -l)"; done
mb() {  # <tree> <tag> <args...>
  local t=$1 tag=$2; shift 2
  ( cd $M/w/mep_$t && PYTHONPATH=. timeout 1800 torchrun --nproc_per_node=4 --master_port=$((29600 + RANDOM % 400)) \
      $K/microbench.py --out $O/mb_${tag}.json "$@" > $O/mb_${tag}.log 2>&1 )
  note "mb $tag rc=$? $(python3 -c "import json;d=json.load(open('$O/mb_${tag}.json'));print('ms',[round(x,2) for x in d['median_ms_per_rank']],'GiB',[round(x,2) for x in d['peak_gib_per_rank']])" 2>/dev/null)"
}
for t in old c5s; do
  mb $t experts_uniform_$t --mode experts --routing uniform
  mb $t experts_skew_$t --mode experts --routing skew
  mb $t moe_$t --mode moe
done
cell() {  # <name> <config> <cache> <steps>
  local D=$O/$1; rm -rf $D; mkdir -p $D
  ( cd $M/w/mep_c5s && PROBE_STEPS=$4 PROBE_DET=1 PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$3/ic TRITON_CACHE_DIR=$3/tc \
    NGPU=4 LOG_RANK=0,1,2,3 MODULE=moonep_probe_1003 CONFIG=$2 timeout 1800 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out; note "$1 rc=$rc"
}
CELLS="standard moonep moonep_full standard_ep2 moonep_ep2 standard_hsdp moonep_hsdp"
C=$O/cache; rm -rf $C; mkdir -p $C
for c in $CELLS; do cell warm_$c ${c}_cell $C 1; done
for c in $CELLS; do rm -rf $O/cc; cp -r $C $O/cc; cell num_$c ${c}_cell $O/cc 20; done
rm -rf $O/cc; cp -r $C $O/cc; cell num_standard_b standard_cell $O/cc 20
rm -rf $O/cc $C
note "all done"; echo done > $O/DONE
