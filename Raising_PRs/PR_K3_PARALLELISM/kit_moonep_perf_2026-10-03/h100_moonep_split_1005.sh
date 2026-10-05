#!/bin/bash
# 10-05, moonep_review1 = f210edd3a (c3, the shared-experts stream, moved to its own branch; c4 and c5 replayed on c2):
#   1. GPU tests on the new head (test_moonep.py, test_moonep_ops.py), real MoonEP;
#   2. bitwise dumps, the old head 0b698e3c2 against the new one, the six scenarios: must be bitwise equal;
#   3. item 12: step-1 full gradients of standard EP and MoonEP at FSDP 4 x EP 4 (efsdp 1), dp_shard 4 x EP 2 (efsdp 2)
#      and HSDP 2 x 2 x EP 2, plus standard twice; every dump run on a copy of one cache warmed by all six configs.
M=~/mep; V=$M/venv_src; K=~/kit/kit_moonep_perf_2026-10-03
O=$M/results/moonep_split; mkdir -p $O; P=$O/progress.txt
export PYTORCH_ALLOC_CONF=expandable_segments:True PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
note() { echo "$(date +%H:%M:%S) $*" >> $P; }
. $V/bin/activate
git -C $M/tt fetch -q origin moonep_review1
W=$M/w/mep_c5s; [ -d $W ] || git -C $M/tt worktree add -q --detach $W f210edd3a
for t in mep_c5 mep_c5s; do note "$t $(git -C $M/w/$t rev-parse --short HEAD) dirty=$(git -C $M/w/$t status --short | wc -l)"; done
( cd $W && timeout 2400 python -m pytest tests/unit_tests/gpu/test_moonep.py tests/unit_tests/cpu/test_moonep_ops.py -q -rA > $O/gpu_tests.log 2>&1 )
note "gpu tests rc=$? $(tail -1 $O/gpu_tests.log)"

R=$O/bitwise; mkdir -p $R
for sc in "2 1 1 hot none" "2 2 2 hot none" "2 2 2 hot selective" "2 2 2 hot full" "2 2 2 hot region" "4 2 2 hot2 none"; do
  set -- $sc; n=$1; name="n$1_l$2_m$3_$4_$5"
  for side in a:mep_c5 b:mep_c5s; do
    T=$M/w/${side#*:}
    PYTHONPATH=$T CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((n - 1))) timeout 900 torchrun --nproc_per_node=$n \
      --master_port=$((29600 + RANDOM % 400)) $K/bitwise_dump.py --out $R/${side%%:*}/$name --layers $2 --mbs $3 \
      --routing $4 --ac $5 > $R/${side%%:*}.$name.log 2>&1
  done
  python $K/bitwise_cmp.py $R/a/$name $R/b/$name > $R/cmp.$name.txt 2>&1
  note "bitwise c5 vs c5s $name: $(tail -1 $R/cmp.$name.txt)"
done

G=$O/grads; mkdir -p $G
cell() {  # <name> <config> <cache> <dump file or empty>
  local D=$O/$1; rm -rf $D; mkdir -p $D
  ( cd $W && PROBE_STEPS=1 PROBE_DET=1 PROBE_GRAD_OUT=$4 PYTHONPATH=$K:. TORCHINDUCTOR_CACHE_DIR=$3/ic \
    TRITON_CACHE_DIR=$3/tc NGPU=4 LOG_RANK=0,1,2,3 MODULE=moonep_grad_probe_1005 CONFIG=$2 timeout 1800 \
    ./run_train.sh --output-dir $D/out > $D/run.log 2>&1 )
  local rc=$?; rm -rf $D/out; note "$1 rc=$rc $(ls -la $4 2>/dev/null | awk '{print $5}')"
}
C=$O/cache; rm -rf $C; mkdir -p $C
for c in standard moonep standard_ep2 moonep_ep2 standard_hsdp moonep_hsdp; do cell warm_$c ${c}_cell $C ""; done
for c in standard moonep standard_ep2 moonep_ep2 standard_hsdp moonep_hsdp; do
  rm -rf $O/cc; cp -r $C $O/cc; cell dump_$c ${c}_cell $O/cc $G/$c.pt
done
rm -rf $O/cc; cp -r $C $O/cc; cell dump_standard_b standard_cell $O/cc $G/standard_b.pt
rm -rf $O/cc $C
python $K/cmp_grads_1005.py $G > $O/cmp_grads.txt 2>&1
note "grads: $(grep -c parameters $O/cmp_grads.txt) pairs compared"
note "all done"; echo done > $O/DONE
