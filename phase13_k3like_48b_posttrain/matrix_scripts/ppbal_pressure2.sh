#!/bin/bash
# pp2 on the 32-layer flavor: 16 layers per rank, so what autograd saves per
# microbatch is 16 layers deep; tokens per microbatch raised until the cards
# are nearly full. Then balancing from the heavier rank onto the lighter one.
set -uo pipefail
TITAN=${TITAN:-/tmp/wt_ppport}
OUT=/workspace/ppbal_pressure2_$(date +%m%d_%H%M%S); mkdir -p $OUT; R=$OUT/results.txt; : > $R
COMMON="--module kimi_k3 --debug.seed 42 --debug.deterministic --training.steps 6 --metrics.log_freq 1 --parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 2 --parallelism.pipeline-parallel-layers-per-stage 16 --parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0 --parallelism.pipeline_parallel_schedule 1F1B --training.max_context_length 256"
arm() { local name=$1 cfg=$2 tok=$3; shift 3
  local d=$OUT/$name; mkdir -p $d
  ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i 0,1 >> $d/mem.csv; sleep 1; done ) & local sp=$!
  ( source /venv/main/bin/activate && cd $TITAN && CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$TITAN timeout 2400 torchrun --nproc_per_node=2 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train $COMMON --config $cfg --training.num-tokens-per-microbatch-per-dp-rank $tok --training.num-tokens-per-train-step $((tok*8)) --dump-folder $d "$@" > $d/run.log 2>&1 ); local rc=$?
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  local peaks; peaks=$(awk -F', ' '{ if ($2+0 > m[$1]) m[$1]=$2+0 } END { for (i=0;i<2;i++) printf "%d:%.2f ", i, m[i]/1024 }' $d/mem.csv)
  local s1 s6; s1=$(grep -oE "step: *1 .*loss: *[0-9.]+" $d/run.log | head -1 | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+'); s6=$(grep -oE "step: *6 .*loss: *[0-9.]+" $d/run.log | head -1 | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+')
  local oom; oom=$(grep -c "OutOfMemoryError" $d/run.log)
  printf "%-14s tok/mb=%-5s rc=%s oom=%s s1=%-9s s6=%-9s peakGiB[%s]\n" "$name" "$tok" "$rc" "$oom" "$s1" "$s6" "$peaks" >> $R; tail -1 $R
  echo "$peaks" > $d/peaks.txt
}
for tok in 2048 4096 8192; do arm base$tok kimi_k3_debugmodel_32l $tok; done
TOK=""; for tok in 8192 4096 2048; do grep -q "base$tok .*rc=0 oom=0" $R && { TOK=$tok; break; }; done
[ -z "$TOK" ] && { echo "no baseline survived" >> $R; exit 1; }
P=$(cat $OUT/base$TOK/peaks.txt); read A B <<< "$(echo $P | sed -E 's/0:([0-9.]+) 1:([0-9.]+) /\1 \2/')"
if python3 -c "import sys; sys.exit(0 if float('$A') >= float('$B') else 1)"; then SRC=0; DST=1; else SRC=1; DST=0; fi
echo "balancing at tok/mb=$TOK: source=$SRC dest=$DST (peaks 0:$A 1:$B)" | tee -a $R
K3_PPBAL_SRC=$SRC K3_PPBAL_DST=$DST K3_PPBAL_POOL_GIB=5 arm ppbal$TOK kimi_k3_debugmodel_32l_ppbal $TOK
grep -aciE "pool|park|balance" $OUT/ppbal$TOK/run.log | xargs -I{} echo "balance log lines: {}" >> $R
echo "DONE $OUT" >> $R
