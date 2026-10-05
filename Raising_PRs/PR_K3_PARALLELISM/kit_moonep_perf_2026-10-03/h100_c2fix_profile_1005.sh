#!/bin/bash
# 10-05: the c2 fix (edb3f6f88 on moonep_review1, on top of 0b698e3c2; c2fix_on_c2.patch is the same edit on c2 alone)
# on the 4 x H100: GPU tests on the new head, bitwise 0b698e3c2 vs edb3f6f88 on real MoonEP, the expert-op microbench for
# c1 / c2 / c2+fix / c5 / c6 and the MoE layer for c5 / c6, then per-kernel profiler tables for c1, c2 and c2+fix.
M=~/mep; V=$M/venv_src; K=~/kit/kit_moonep_perf_2026-10-03
O=$M/results/c2fix; mkdir -p $O; P=$O/progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
git -C $M/tt fetch -q origin moonep_review1
[ -d $M/w/mep_c2fix ] || { git -C $M/tt worktree add -q --detach $M/w/mep_c2fix 06d8c338d && git -C $M/w/mep_c2fix apply $K/c2fix_on_c2.patch; }
[ -d $M/w/mep_c6 ] || git -C $M/tt worktree add -q --detach $M/w/mep_c6 edb3f6f88
for t in c1 c2 c2fix c5 c6; do note "$t $(git -C $M/w/mep_$t rev-parse --short HEAD) dirty=$(git -C $M/w/mep_$t status --short | wc -l)"; done
( cd $M/w/mep_c6 && timeout 2400 python -m pytest tests/unit_tests/gpu/test_moonep.py tests/unit_tests/cpu/test_moonep_ops.py \
    tests/unit_tests/gpu/test_moe_shared_experts_stream.py -q -rA > $O/gpu_tests.log 2>&1 )
note "gpu tests c6 rc=$? $(tail -1 $O/gpu_tests.log)"
R=$O/bitwise; mkdir -p $R
for sc in "2 1 1 hot none" "2 2 2 hot none" "2 2 2 hot selective" "2 2 2 hot full" "2 2 2 hot region" "4 2 2 hot2 none"; do
  set -- $sc; n=$1; name="n$1_l$2_m$3_$4_$5"
  for side in a:mep_c5 b:mep_c6; do
    T=$M/w/${side#*:}
    PYTHONPATH=$T CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) timeout 900 torchrun --nproc_per_node=$n \
      --master_port=$((29600 + RANDOM % 400)) $K/bitwise_dump.py --out $R/${side%%:*}/$name --layers $2 --mbs $3 \
      --routing $4 --ac $5 > $R/${side%%:*}.$name.log 2>&1
  done
  python $K/bitwise_cmp.py $R/a/$name $R/b/$name > $R/cmp.$name.txt 2>&1
  note "bitwise c5 vs c6 $name: $(tail -1 $R/cmp.$name.txt)"
done
mb() {  # <tree> <tag> <args...>
  local t=$1 tag=$2; shift 2
  ( cd $M/w/mep_$t && PYTHONPATH=. timeout 1800 torchrun --nproc_per_node=4 --master_port=$((29600 + RANDOM % 400)) \
      $K/microbench.py --out $O/mb_${tag}.json "$@" > $O/mb_${tag}.log 2>&1 )
  note "mb $tag rc=$? $(python3 -c "import json;d=json.load(open('$O/mb_${tag}.json'));print('ms',[round(x,2) for x in d['median_ms_per_rank']],'GiB',[round(x,2) for x in d['peak_gib_per_rank']])" 2>/dev/null)"
}
for t in c1 c2 c2fix c5 c6; do mb $t experts_uniform_$t --mode experts --routing uniform; done
for t in c5 c6; do mb $t moe_$t --mode moe; done
for t in c1 c2 c2fix; do mb $t prof_$t --mode experts --routing uniform --iters 3 --warmup 3 --profile $O/prof_$t.txt; done
note "all done"; echo done > $O/DONE
