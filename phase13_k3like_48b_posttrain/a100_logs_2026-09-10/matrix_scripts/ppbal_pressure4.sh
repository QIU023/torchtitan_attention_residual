#!/bin/bash
# pp2 on the 32-layer flavor: 16 layers per rank, so what autograd saves per
# microbatch is 16 layers deep; tokens per microbatch raised until the cards
# are nearly full. Then balancing from the heavier rank onto the lighter one.
set -uo pipefail
TITAN=${TITAN:-/tmp/wt_ppport}
OUT=/workspace/ppbal_pressure4_$(date +%m%d_%H%M%S); mkdir -p $OUT; R=$OUT/results.txt; : > $R
COMMON="--module kimi_k3 --debug.seed 42 --debug.deterministic --training.steps 6 --metrics.log_freq 1 --parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 4 --parallelism.pipeline-parallel-layers-per-stage 8 --parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0 --parallelism.pipeline_parallel_schedule 1F1B --training.max_context_length 256"
arm() { local name=$1 cfg=$2 tok=$3; shift 3
  local d=$OUT/$name; mkdir -p $d
  ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits -i 0,1,2,3 >> $d/mem.csv; sleep 1; done ) & local sp=$!
  ( source /venv/main/bin/activate && cd $TITAN && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONPATH=$TITAN timeout 2400 torchrun --nproc_per_node=4 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train $COMMON --config $cfg --training.num-tokens-per-microbatch-per-dp-rank $tok --training.num-tokens-per-train-step $((tok*8)) --dump-folder $d "$@" > $d/run.log 2>&1 ); local rc=$?
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  local peaks; peaks=$(awk -F', ' '{ if ($2+0 > m[$1]) m[$1]=$2+0 } END { for (i=0;i<4;i++) printf "%d:%.2f ", i, m[i]/1024 }' $d/mem.csv)
  local s1 s6; s1=$(grep -oE "step: *1 .*loss: *[0-9.]+" $d/run.log | head -1 | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+'); s6=$(grep -oE "step: *6 .*loss: *[0-9.]+" $d/run.log | head -1 | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+')
  local oom; oom=$(grep -c "OutOfMemoryError" $d/run.log)
  printf "%-14s tok/mb=%-5s rc=%s oom=%s s1=%-9s s6=%-9s peakGiB[%s]\n" "$name" "$tok" "$rc" "$oom" "$s1" "$s6" "$peaks" >> $R; tail -1 $R
  echo "$peaks" > $d/peaks.txt
}
for tok in 2048 4096; do arm base$tok kimi_k3_debugmodel_32l $tok; done
TOK=""; for tok in 4096 2048; do grep -q "base$tok .*rc=0 oom=0" $R && { TOK=$tok; break; }; done
[ -z "$TOK" ] && { echo "no baseline survived" >> $R; exit 1; }
echo "balancing at tok/mb=$TOK: sources=0,1 dest=3" | tee -a $R
K3_PPBAL_SRC=0,1 K3_PPBAL_DST=3 K3_PPBAL_POOL_GIB=5 arm ppbal${TOK}_src01 kimi_k3_debugmodel_32l_ppbal $TOK
for a in base$TOK ppbal${TOK}_src01; do echo "--- $a active peaks:" >> $R; grep -ah "PPBAL.*stats\|max_active\|max active" $OUT/$a/run.log | head -4 | cut -c1-160 >> $R; done
echo "DONE $OUT" >> $R
