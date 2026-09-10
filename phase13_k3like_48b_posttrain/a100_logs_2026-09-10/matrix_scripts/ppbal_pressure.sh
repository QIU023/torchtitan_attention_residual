#!/bin/bash
# pp8 x vp4 on the 32-layer flavor with the per-microbatch token count raised until
# the 16 GB cards are nearly full; per-GPU peak memory sampled from nvidia-smi.
# Arms: baseline at each token count, then balancing with the two heaviest ranks
# parking on the lightest. Loss rows are recorded but the claim here is memory.
set -uo pipefail
TITAN=${TITAN:-/tmp/wt_ppport}
OUT=/workspace/ppbal_pressure_$(date +%m%d_%H%M%S); mkdir -p $OUT; R=$OUT/results.txt; : > $R
COMMON="--module kimi_k3 --debug.seed 42 --debug.deterministic --training.steps 6 --metrics.log_freq 1 --parallelism.data_parallel_shard_degree 1 --parallelism.pipeline_parallel_degree 8 --parallelism.pipeline-parallel-layers-per-stage 1 --parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_first_stage_less_layers 0 --parallelism.pipeline_parallel_last_stage_less_layers 0 --parallelism.pipeline_parallel_schedule Interleaved1F1B --training.max_context_length 256"
arm() { local name=$1 cfg=$2 tok=$3; shift 3
  local d=$OUT/$name; mkdir -p $d
  ( while true; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> $d/mem.csv; sleep 1; done ) & local sp=$!
  ( source /venv/main/bin/activate && cd $TITAN && PYTHONPATH=$TITAN timeout 2400 torchrun --nproc_per_node=8 --master_port=$((30000+RANDOM%20000)) -m torchtitan.train $COMMON --config $cfg --training.num-tokens-per-microbatch-per-dp-rank $tok --training.num-tokens-per-train-step $((tok*8)) --dump-folder $d "$@" > $d/run.log 2>&1 ); local rc=$?
  kill $sp 2>/dev/null; wait $sp 2>/dev/null
  local peaks; peaks=$(awk -F', ' '{ if ($2+0 > m[$1]) m[$1]=$2+0 } END { for (i=0;i<8;i++) printf "%d:%.1f ", i, m[i]/1024 }' $d/mem.csv)
  local s1 s6; s1=$(grep -oE "step: *1 .*loss: *[0-9.]+" $d/run.log | head -1 | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+'); s6=$(grep -oE "step: *6 .*loss: *[0-9.]+" $d/run.log | head -1 | grep -oE 'loss: *[0-9.]+' | grep -oE '[0-9.]+')
  local oom; oom=$(grep -c "OutOfMemoryError" $d/run.log)
  printf "%-14s tok/mb=%-5s rc=%s oom=%s s1=%-9s s6=%-9s peakGiB[%s]\n" "$name" "$tok" "$rc" "$oom" "$s1" "$s6" "$peaks" >> $R; tail -1 $R
  echo "$peaks" > $d/peaks.txt
}
pick_src_dst() { # from a peaks line "0:x 1:y ..." -> top-2 as sources, min as dest
  python3 - "$1" <<'PY'
import sys
pairs=[(int(a),float(b)) for a,b in (t.split(':') for t in sys.argv[1].split())]
order=sorted(pairs,key=lambda p:-p[1]); dest=min(pairs,key=lambda p:p[1])[0]
print(f"{order[0][0]},{order[1][0]} {dest}")
PY
}
arm base512 kimi_k3_debugmodel_32l 512
arm base1024 kimi_k3_debugmodel_32l 1024
# choose the heaviest non-OOM baseline
if grep -q "base1024 .*oom=0" $R; then TOK=1024; P=$(cat $OUT/base1024/peaks.txt); else TOK=512; P=$(cat $OUT/base512/peaks.txt); fi
read SRC DST <<< "$(pick_src_dst "$P")"
echo "balancing at tok/mb=$TOK: sources=$SRC dest=$DST" | tee -a $R
K3_PPBAL_SRC=$SRC K3_PPBAL_DST=$DST K3_PPBAL_POOL_GIB=4 arm ppbal$TOK kimi_k3_debugmodel_32l_ppbal $TOK
grep -c "pp_balance\|PPBalance\|parked" $OUT/ppbal$TOK/run.log | xargs -I{} echo "balance log lines: {}" >> $R
echo "DONE $OUT" >> $R
