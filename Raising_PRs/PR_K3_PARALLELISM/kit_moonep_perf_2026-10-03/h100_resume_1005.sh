#!/bin/bash
# 10-05 resume on the same box (the 10-04 run stopped during num_old_moonep): the DEP NCCL unit test on 8518f473f first,
# then the MoonEP end-to-end cells from num_old_moonep on the surviving cache_old, then c5 and c5x as in
# h100_moonep_perf.sh, then the c1 / c2 expert-op microbench once more (c2 was slower than c1 on 10-04).
M=~/mep; V=$M/venv_src; K=~/kit/kit_moonep_perf_2026-10-03
O=$M/results/moonep_perf; P=$O/progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
note "resume 10-05"

G=$M/results/dep_fig11_gputest; rm -rf $G; mkdir -p $G
( cd $M/w/dep_fig11 && echo "$(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)" > $G/tree.txt && \
  timeout 1800 python -m pytest tests/unit_tests/gpu/test_kimi_k3_vision_dep.py -q -rA > $G/pytest.log 2>&1; echo "rc=$?" > $G/rc )
note "dep gpu test $(cat $G/tree.txt) $(cat $G/rc) $(tail -1 $G/pytest.log)"; echo done > $G/DONE

mb() {  # <tree> <tag> <args...>
  local t=$1 tag=$2; shift 2
  ( cd $M/w/mep_$t && PYTHONPATH=. timeout 1800 torchrun --nproc_per_node=4 --master_port=$((29600 + RANDOM % 400)) \
      $K/microbench.py --out $O/mb_${tag}.json "$@" > $O/mb_${tag}.log 2>&1 )
  note "mb $tag rc=$? $(python3 -c "import json;d=json.load(open('$O/mb_${tag}.json'));print('ms',[round(x,2) for x in d['median_ms_per_rank']],'GiB',[round(x,2) for x in d['peak_gib_per_rank']])" 2>/dev/null)"
}
cell() {  # <tree> <name> <config> <cache> <steps>
  local T=$M/w/mep_${1%x} name=$2 cfg=$3 cache=$4 D=$O/$2; rm -rf $D; mkdir -p $D
  ( cd $T && PROBE_STEPS=$5 PROBE_DET=1 PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$cache/ic TRITON_CACHE_DIR=$cache/tc \
    NGPU=4 LOG_RANK=0,1,2,3 MODULE=moonep_probe_1003 CONFIG=$cfg timeout 1800 ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out; note "$name rc=$rc"
}
for c in moonep moonep_full; do rm -rf $O/cache_c; cp -r $O/cache_old $O/cache_c; cell old num_old_$c ${c}_cell $O/cache_c 20; done
rm -rf $O/cache_c; cp -r $O/cache_old $O/cache_c; cell old num_old_standard_b standard_cell $O/cache_c 20
rm -rf $O/cache_c $O/cache_old
for t in c5 c5x; do
  cells="moonep moonep_full moonep_shared_stream standard_shared_stream"
  [ $t = c5x ] && cells="standard_ep2 moonep_ep2 standard_hsdp moonep_hsdp"
  C0=$O/cache_$t; rm -rf $C0; mkdir -p $C0
  for c in $cells; do cell $t warm_${t}_$c ${c}_cell $C0 1; done
  for c in $cells; do rm -rf $O/cache_c; cp -r $C0 $O/cache_c; cell $t num_${t}_$c ${c}_cell $O/cache_c 20; done
  rm -rf $O/cache_c $C0
done
note "e2e done"
mb c1 experts_uniform_c1_r2 --mode experts --routing uniform
mb c2 experts_uniform_c2_r2 --mode experts --routing uniform
note "all done"; echo done > $O/DONE
