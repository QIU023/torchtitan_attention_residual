#!/bin/bash
# A backward-only float32-level perturbation: the expert products keep their float32 forward (so the routing
# is untouched) and take their backward from float64 rounded once; against the float32 dp1 dump.
# Layer 0 at a percent means the backward amplifies float32 rounding a hundredfold.
set -uo pipefail
T=/tmp/wt_ppprobe33x; SEED=$(ls -d /workspace/.mx3_seeds_main33/kimi_k3_debugmodel_*/seed_ckpt | head -1)
OUT=/workspace/ppprobe33v_$(date +%m%d_%H%M); mkdir -p $OUT; DUMP=$OUT/dumps; mkdir -p $DUMP; cp /workspace/ppprobe33y_0904_2307/dumps/dp1.rank0.pt $DUMP/
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
d=$OUT/dp1_bwd64; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
( source /venv/main/bin/activate && cd $T && CUDA_VISIBLE_DEVICES=7 PYTHONPATH=$T TORCHINDUCTOR_CACHE_DIR=$OUT/ind TRITON_CACHE_DIR=$OUT/tri \
  GRAD_TENSOR_DUMP=$DUMP/dp1_bwd64 GRAD_TENSOR_DUMP_EXIT=1 EXPERTS_FP32=1 EXPERTS_BWD_FP64=1 timeout 2400 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
  -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 1 $B \
  --checkpoint.enable --checkpoint.interval 100000 --parallelism.data_parallel_shard_degree 1 --dump-folder $d > $OUT/dp1_bwd64.log 2>&1 ); rm -rf $d/checkpoint
printf "dp1_bwd64 rc=%s dumps=%s\n" $? "$(ls $DUMP/dp1_bwd64.rank*.pt 2>/dev/null | wc -l)"; sed 's/\x1b\[[0-9;]*m//g' $OUT/dp1_bwd64.log | grep -m1 -iE "Traceback|Error" | cut -c1-160
source /venv/main/bin/activate
python - "$DUMP" <<'PY'
import glob, sys, torch
D = sys.argv[1]
def load(prefix):
    d = {}
    for f in sorted(glob.glob(f"{D}/{prefix}.rank*.pt")): d.update(torch.load(f, map_location="cpu"))
    return d
ga, gb = load("dp1"), load("dp1_bwd64"); rels = []; by_layer = {}
for k in ga:
    if k not in gb: continue
    x, y = ga[k].double(), gb[k].double(); r = ((x - y).norm() / (x.norm() + 1e-30)).item(); rels.append(r)
    if k.startswith("layers."):
        li = int(k.split(".")[1]); b = by_layer.setdefault(li, [0.0, 0.0]); b[0] += float((x - y).pow(2).sum()); b[1] += float(x.pow(2).sum())
rels.sort()
print(f"### dp1 float32 experts vs dp1 with the expert BACKWARD rounded from float64 (forward unchanged): {len(rels)} parameters; rel L2 diff median {rels[len(rels)//2]:.2e} p90 {rels[int(0.9*len(rels))]:.2e} max {rels[-1]:.2e}; identical {sum(1 for r in rels if r == 0)}")
print("  by layer: " + " ".join(f"{li}:{((d2 / s2) ** 0.5 if s2 else 0):.0e}" for li, (d2, s2) in sorted(by_layer.items())))
PY
rm -rf $OUT/ind $OUT/tri
echo "PPPROBE33V DONE $OUT"
