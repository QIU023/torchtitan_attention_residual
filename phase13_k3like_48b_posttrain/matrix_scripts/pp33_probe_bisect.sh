#!/bin/bash
# Where does the pipeline's backward part from one GPU, in float32 with float32 experts?
# pp2 with one stage per rank (1F1B, one boundary), pp2 x vp2 (interleaved, the store in play),
# pp2 x vp4 naive (no delta, no store); each against the dp1 dump of the allocator probe.
set -uo pipefail
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
T=/tmp/wt_ppprobe33x; SEED=$(ls -d /workspace/.mx3_seeds_main33/kimi_k3_debugmodel_*/seed_ckpt | head -1)
OUT=/workspace/ppprobe33z_$(date +%m%d_%H%M); mkdir -p $OUT; DUMP=$OUT/dumps; mkdir -p $DUMP; cp /workspace/ppprobe33y_0904_2307/dumps/dp1.rank0.pt $DUMP/
D="--parallelism.data_parallel_shard_degree 1"; P="--parallelism.pipeline_parallel_degree"
L="--parallelism.pipeline-parallel-layers-per-stage"; MB="--parallelism.num-pp-microbatches 8"
B="--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 --training.dtype float32"
run(){ local nm=$1 np=$2 devs=$3 cfg=$4; shift 4
  local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r --reflink=auto $SEED $d/checkpoint
  ( source /venv/main/bin/activate && cd $T && CUDA_VISIBLE_DEVICES=$devs PYTHONPATH=$T TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
    GRAD_TENSOR_DUMP=$DUMP/$nm GRAD_TENSOR_DUMP_EXIT=1 EXPERTS_FP32=1 timeout 2400 torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
    -m torchtitan.train --module kimi_k3 --config $cfg --debug.seed 42 --debug.deterministic \
    --metrics.log_freq 1 --training.steps 1 $B --checkpoint.enable --checkpoint.interval 100000 $D "$@" \
    --dump-folder $d > $OUT/$nm.log 2>&1 ); rm -rf $d/checkpoint
  printf "%-14s rc=%s dumps=%s\n" $nm $? "$(ls $DUMP/$nm.rank*.pt 2>/dev/null | wc -l)"
  sed 's/\x1b\[[0-9;]*m//g' $OUT/$nm.log | grep -m1 -iE "Traceback|Error|OutOfMemory" | cut -c1-160
}
run pp2_1f1b 2 6,7 kimi_k3_debugmodel $P 2 $MB --parallelism.pipeline_parallel_schedule 1F1B
run pp2_vp2 2 6,7 kimi_k3_debugmodel $P 2 $L 9 $MB --parallelism.pipeline_parallel_schedule Interleaved1F1B
run pp2_vp4_naive 2 6,7 kimi_k3_debugmodel_pp_naive $P 2 $L 4 $MB --parallelism.pipeline_parallel_schedule Interleaved1F1B
source /venv/main/bin/activate
python - "$DUMP" <<'PY'
import glob, sys, torch
D = sys.argv[1]
def load(prefix):
    d = {}
    for f in sorted(glob.glob(f"{D}/{prefix}.rank*.pt")): d.update(torch.load(f, map_location="cpu"))
    return d
def group(name):
    for key, g in (("tok_embeddings", "embedding"), ("lm_head", "head"), ("output_res", "output_res"), ("experts", "experts"), ("router", "router"),
                   ("shared", "shared experts"), ("delta_attention", "kda"), ("attention", "attention"), ("norm", "norms")):
        if key in name: return g
    return "other"
ga = load("dp1")
for other in ("pp2_1f1b", "pp2_vp2", "pp2_vp4_naive"):
    gb = load(other)
    if not gb: print(f"### {other}: no dumps"); continue
    rels = []; groups = {}; by_layer = {}
    for k in ga:
        if k not in gb: continue
        x, y = ga[k].double(), gb[k].double(); r = ((x - y).norm() / (x.norm() + 1e-30)).item(); rels.append((r, k))
        g = groups.setdefault(group(k), [0.0, 0.0]); g[0] += float((x - y).pow(2).sum()); g[1] += float(x.pow(2).sum())
        if k.startswith("layers."):
            li = int(k.split(".")[1]); b = by_layer.setdefault(li, [0.0, 0.0]); b[0] += float((x - y).pow(2).sum()); b[1] += float(x.pow(2).sum())
    rels.sort(); rs = [r for r, _ in rels]
    print(f"### float32 + float32 experts, dp1 vs {other}: {len(rs)} parameters; rel L2 diff median {rs[len(rs)//2]:.2e} p90 {rs[int(0.9*len(rs))]:.2e} max {rs[-1]:.2e}; identical {sum(1 for r in rs if r == 0)}")
    print("  groups: " + ", ".join(f"{g} {((d2 / s2) ** 0.5 if s2 else 0):.1e}" for g, (d2, s2) in sorted(groups.items())))
    print("  by layer: " + " ".join(f"{li}:{((d2 / s2) ** 0.5 if s2 else 0):.0e}" for li, (d2, s2) in sorted(by_layer.items())))
PY
rm -rf $OUT/ind_* $OUT/tri_*
echo "PPPROBE33Z DONE $OUT"
