#!/bin/bash
# 10-05, PR head = moonep_review1 = 6e3de1b8d (c2 with the fix folded in; tree equal to edb3f6f88) on 4 x H100:
#   A. item 12: step-1 full gradients of standard EP and MoonEP at FSDP 4 x EP 4, dp_shard 4 x EP 2 and HSDP 2 x 2 x EP 2,
#      plus standard twice, every dump on a copy of one cache warmed by the six configs; cmp_grads_1005.py;
#   B. the body's end-to-end cells on the head, 20 steps: standard (twice), MoonEP, MoonEP FullAC, and standard and
#      MoonEP at dp_shard 4 x EP 2 and HSDP, one cache warmed by every cell;
#   C. microbench, the old PR head 5e4596dc7 and the new head in one session: expert op (uniform, skew), MoE layer.
M=~/mep; V=$M/venv_src; K=~/kit/kit_moonep_perf_2026-10-03
O=$M/results/moonep_final; mkdir -p $O; P=$O/progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
git -C $M/tt fetch -q origin moonep_review1
W=$M/w/mep_head; [ -d $W ] || git -C $M/tt worktree add -q --detach $W 6e3de1b8d
for t in mep_old mep_head; do note "$t $(git -C $M/w/$t rev-parse --short HEAD) dirty=$(git -C $M/w/$t status --short | wc -l)"; done
cell() {  # <name> <module> <config> <cache> <steps> <dump file or empty>
  local D=$O/$1; rm -rf $D; mkdir -p $D
  ( cd $W && PROBE_STEPS=$5 PROBE_DET=1 PROBE_GRAD_OUT=$6 PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$4/ic \
    TRITON_CACHE_DIR=$4/tc NGPU=4 LOG_RANK=0,1,2,3 MODULE=$2 CONFIG=$3 timeout 1800 \
    ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out; note "$1 rc=$rc"
}
G=$O/grads; mkdir -p $G; C=$O/cache_g; rm -rf $C; mkdir -p $C
for c in standard moonep standard_ep2 moonep_ep2 standard_hsdp moonep_hsdp; do cell gwarm_$c moonep_grad_probe_1005 ${c}_cell $C 1 ""; done
for c in standard moonep standard_ep2 moonep_ep2 standard_hsdp moonep_hsdp; do
  rm -rf $O/cc; cp -r $C $O/cc; cell dump_$c moonep_grad_probe_1005 ${c}_cell $O/cc 1 $G/$c.pt
done
rm -rf $O/cc; cp -r $C $O/cc; cell dump_standard_b moonep_grad_probe_1005 standard_cell $O/cc 1 $G/standard_b.pt
rm -rf $O/cc $C
python $K/cmp_grads_1005.py $G > $O/cmp_grads.txt 2>&1
note "item 12 done: $(head -1 $O/cmp_grads.txt | cut -c1-120)"
CELLS="standard moonep moonep_full standard_ep2 moonep_ep2 standard_hsdp moonep_hsdp"
C=$O/cache_e; rm -rf $C; mkdir -p $C
for c in $CELLS; do cell warm_$c moonep_probe_1003 ${c}_cell $C 1 ""; done
for c in $CELLS; do rm -rf $O/cc; cp -r $C $O/cc; cell num_$c moonep_probe_1003 ${c}_cell $O/cc 20 ""; done
rm -rf $O/cc; cp -r $C $O/cc; cell num_standard_b moonep_probe_1003 standard_cell $O/cc 20 ""
rm -rf $O/cc $C
note "e2e done"
mb() {  # <tree> <tag> <args...>
  local t=$1 tag=$2; shift 2
  ( cd $M/w/mep_$t && PYTHONPATH=. timeout 1800 torchrun --nproc_per_node=4 --master_port=$((29600 + RANDOM % 400)) \
      $K/microbench.py --out $O/mb_${tag}.json "$@" > $O/mb_${tag}.log 2>&1 )
  note "mb $tag rc=$? $(python3 -c "import json;d=json.load(open('$O/mb_${tag}.json'));print('ms',[round(x,2) for x in d['median_ms_per_rank']],'GiB',[round(x,2) for x in d['peak_gib_per_rank']])" 2>/dev/null)"
}
for t in old head; do
  mb $t experts_uniform_$t --mode experts --routing uniform
  mb $t experts_skew_$t --mode experts --routing skew
  mb $t moe_$t --mode moe
done
note "all done"; echo done > $O/DONE
