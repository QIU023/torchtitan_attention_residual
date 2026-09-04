#!/bin/bash
# The exact test of the pipeline's gradients: the 33-layer model with --training.dtype float32
# (parameters, activations and the gradient accumulation in float32; the grouped expert GEMM
# stays bf16 inside, with identical inputs per micro-batch), dp1 against pp2 x vp4 and pp8 x vp4,
# step-1 gradients dumped in float32 and the run left before the optimizer step.
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
MS=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts
T=/tmp/wt_ppprobe33f; SEED=$(ls -d /workspace/.mx3_seeds_main33/kimi_k3_debugmodel_*/seed_ckpt | head -1)
OUT=/workspace/ppprobe33f_$(date +%m%d_%H%M); mkdir -p $OUT; DUMP=$OUT/dumps; mkdir -p $DUMP
D="--parallelism.data_parallel_shard_degree 1"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"
IL="--parallelism.num-pp-microbatches 8 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
run(){ local nm=$1 np=$2; shift 2
  local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
  ( source /venv/main/bin/activate && cd $T && PYTHONPATH=$T TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
    GRAD_TENSOR_DUMP=$DUMP/$nm GRAD_TENSOR_DUMP_EXIT=1 timeout 2400 torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps 1 $B --checkpoint.enable --checkpoint.interval 100000 $D "$@" \
    --dump-folder $d > $OUT/$nm.log 2>&1 ); rm -rf $d/checkpoint
  printf "%-10s rc=%s %s dumps=%s\n" $nm $? "$(sed 's/\x1b\[[0-9;]*m//g' $OUT/$nm.log | grep -oE 'step: +1 +loss: +[0-9.]+' | head -1)" "$(ls $DUMP/$nm.rank*.pt 2>/dev/null | wc -l)"
  sed 's/\x1b\[[0-9;]*m//g' $OUT/$nm.log | grep -m1 -iE "Traceback|Error|OutOfMemory" | cut -c1-160
}
run dp1 1
run pp2_vp4 2 $P 2 $L 4 $IL
run pp8_vp4 8 $P 8 $L 1 $IL
source /venv/main/bin/activate
python $MS/pp_step10_census.py "float32 model: dp1 vs pp2 x vp4" $DUMP/dp1 $DUMP/pp2_vp4
python $MS/pp_step10_census.py "float32 model: dp1 vs pp8 x vp4" $DUMP/dp1 $DUMP/pp8_vp4
python - <<'PY'
import glob, torch, sys
def load(prefix):
    d = {}
    for f in sorted(glob.glob(prefix + ".rank*.pt")): d.update(torch.load(f, map_location="cpu"))
    return d
base = load(sys.argv[1] if len(sys.argv) > 1 else "/dev/null")
PY
python - "$DUMP" <<'PY'
import glob, sys, torch
D = sys.argv[1]
def load(prefix):
    d = {}
    for f in sorted(glob.glob(prefix + ".rank*.pt")): d.update(torch.load(f, map_location="cpu"))
    return d
a = load(f"{D}/dp1")
for other in ("pp2_vp4", "pp8_vp4"):
    b = load(f"{D}/{other}"); worst = []
    for k in a:
        if k not in b: continue
        x, y = a[k].double(), b[k].double()
        rel = ((x - y).norm() / (x.norm() + 1e-30)).item(); mx = (x - y).abs().max().item()
        worst.append((rel, mx, k))
    worst.sort(reverse=True)
    rels = sorted(w[0] for w in worst)
    print(f"### float32 exact comparison dp1 vs {other}: {len(worst)} parameters; rel L2 diff median {rels[len(rels)//2]:.2e}, p90 {rels[int(0.9*len(rels))]:.2e}, max {rels[-1]:.2e}; worst: " + ", ".join(f"{k} ({r:.1e}, max abs {m:.1e})" for r, m, k in worst[:3]))
PY
rm -rf $DUMP $OUT/ind_* $OUT/tri_*
echo "PPPROBE33F DONE $OUT"
