#!/bin/bash
# Old head (OLD) vs working tree (NEW) on the fake MoonEP: dump each scenario on both trees, then compare.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
P=$S/moonep_perf_1003; OLD=${OLD:-$S/wt_moonep_old_5e45}; NEW=${NEW:-$S/wt_moonep_clean}; TAG=${TAG:-c1}
FK=${FK:-/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_moonep_rewrite_2026-09-29/local/fake_moonep}
. /workspace/venv_0928/bin/activate
R=$P/bitwise_$TAG; rm -rf $R; mkdir -p $R
for sc in "2 1 1 hot none" "2 2 2 hot none" "2 2 2 hot selective" "2 2 2 hot full" "2 2 2 hot region" "4 2 2 hot2 none"; do
  set -- $sc; n=$1; name="n$1_l$2_m$3_$4_$5"
  for tree in old new; do
    T=$OLD; [ $tree = new ] && T=$NEW
    FAKE_MOONEP_ASYNC_SLEEP=${SLEEP:-0} PYTHONPATH=$FK:$T CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) timeout 600 torchrun --nproc_per_node=$n --master_port=$((29600 + RANDOM % 400)) \
      $P/bitwise_dump.py --out $R/$tree/$name --layers $2 --mbs $3 --routing $4 --ac $5 > $R/$tree.$name.log 2>&1
    echo "$name $tree rc=$?" >> $R/progress.txt
  done
  python $P/bitwise_cmp.py $R/old/$name $R/new/$name > $R/cmp.$name.txt 2>&1
  echo "$name: $(tail -1 $R/cmp.$name.txt)" >> $R/progress.txt
done
echo done >> $R/progress.txt
