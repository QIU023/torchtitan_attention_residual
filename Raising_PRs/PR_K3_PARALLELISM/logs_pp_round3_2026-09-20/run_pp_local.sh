#!/bin/bash
# PR 4312 round 3: the C4 100-step cells of the H100 body tables, on the round-3 debug model, on the local
# 8 x RTX 5060 Ti. Reference only (functional, structure of the comparison), never for the body.
# Tree = pp_review4 worktree with probe_apply.py applied. Protocol as the body's: seed 42, deterministic,
# one seed checkpoint per batch shape, fp32 total grad norm (GN_FP32), the reference accumulating like
# the pipeline (NOSYNC_GA) and every other cell on a copy of the reference's warm compile cache.
set -u
TT=/tmp/wt_pp4312
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
OUT=${OUT:-$S/ppmx}; mkdir -p $OUT
source /workspace/venv_bfx9/bin/activate
export PYTHONPATH=$TT GN_FP32=1 STEPS=${STEPS:-100}
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1"  # the c4 flavors carry the checkpointer (load only); the seed flavor sets create_seed_checkpoint
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
B8="--training.num-tokens-per-train-step 2048 --training.num-tokens-per-microbatch-per-dp-rank 256"
D="--parallelism.data_parallel_shard_degree"; E="--parallelism.expert_parallel_degree"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"
P4="--parallelism.pipeline_parallel_degree 4 --parallelism.num-pp-microbatches 4 $IL"
C4=kimi_k3_debugmodel_c4; C4N=kimi_k3_debugmodel_c4_pp_naive; C48=kimi_k3_debugmodel_c4_8stages; C48N=kimi_k3_debugmodel_c4_8stages_naive

seed() {  # seed <tag> <batch flags...>
  local tag=$1; shift
  [ -d $OUT/seed_$tag/checkpoint ] && return
  ( cd $TT && CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR=$OUT/ind_seed TRITON_CACHE_DIR=$OUT/tri_seed torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) \
      $COMMON --config kimi_k3_debugmodel_c4_seed --training.steps 1 "$@" --dump-folder $OUT/seed_$tag > $OUT/seed_$tag.log 2>&1 )
  echo "seed_$tag rc=$?"
}

cell() {  # cell <name> <gpus> <nproc> <seed tag> <cache source or -> <config> <flags...>
  local nm=$1 gpus=$2 np=$3 stag=$4 csrc=$5 cfg=$6; shift 6
  local d=$OUT/$nm; rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d; cp -r $OUT/seed_$stag/checkpoint $d/
  if [ "$csrc" != "-" ] && [ -d $OUT/ind_$csrc ]; then cp -r $OUT/ind_$csrc $OUT/ind_$nm; cp -r $OUT/tri_$csrc $OUT/tri_$nm 2>/dev/null; fi
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm \
      torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) \
      $COMMON --config $cfg --training.steps $STEPS "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; rm -rf $d/checkpoint
  local loaded=$(grep -a -c "Loading the checkpoint" $OUT/$nm.log)
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) seed_loaded=$loaded"
}

table() {  # table <reference> <names...>
  python3 - "$OUT" "$@" <<'PY'
import os, re, sys
out, ref, names = sys.argv[1], sys.argv[2], sys.argv[3:]
STEPS = tuple(int(x) for x in os.environ.get("TABLE_STEPS", "1 10 20 50 100").split())
def series(f):
    L, G = {}, {}
    for line in open(f, errors="ignore"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"step: *(\d+)\s.*loss: *([0-9.-]+).*grad_norm: *([0-9.]+)", line)
        if m and float(m.group(2)) > 0:
            L[int(m.group(1))] = float(m.group(2)); G[int(m.group(1))] = float(m.group(3))
    return L, G
R = series(f"{out}/{ref}.log")
def allsteps(X, Y):
    common = [s for s in Y if s in X]
    same = sum(1 for s in common if X[s] == Y[s]); return f"{same}/{len(common)}"
hdr = " | ".join(f"step {s}" for s in STEPS)
print(f"| cell | {hdr} | {hdr} (grad norm) | steps identical (loss, norm) |")
print("|---" * (2 + 2 * len(STEPS)) + "|")
for nm in [ref] + names:
    try: L, G = series(f"{out}/{nm}.log")
    except FileNotFoundError: print(f"| {nm} | missing |"); continue
    def fmt(X, Y, s):
        if s not in X: return "-"
        if nm == ref or s not in Y: return f"`{X[s]:.6f}`"
        if X[s] == Y[s]: return f"`{X[s]:.6f}` (bitwise)"
        return f"`{X[s]:.6f}` ({(X[s]-Y[s])/Y[s]*100:+.3g}%)"
    ident = "-" if nm == ref else f"{allsteps(L, R[0])}, {allsteps(G, R[1])}"
    print(f"| {nm} | " + " | ".join(fmt(L, R[0], s) for s in STEPS) + " | " + " | ".join(fmt(G, R[1], s) for s in STEPS) + f" | {ident} |")
PY
}

seed c4_1024 $B4
seed c4_2048 $B8
echo "# 1024 tokens per step"
( export NOSYNC_GA=1; cell dp1_ns 0 1 c4_1024 - $C4 $B4 $D 1 )               # the reference, and the warm cache
cell dp1 1 1 c4_1024 dp1_ns $C4 $B4 $D 1 &
( export MB_REVERSE=1; cell dp1_rev 2 1 c4_1024 dp1_ns $C4 $B4 $D 1 ) &
cell pp2 3,4 2 c4_1024 dp1_ns $C4 $B4 $D 1 $P &
wait
cell vp2c 0,1 2 c4_1024 dp1_ns $C4 $B4 $D 1 $P $IL &
cell vp2n 2,3 2 c4_1024 dp1_ns $C4N $B4 $D 1 $P $IL &
cell pp2vp4c 4,5 2 c4_1024 dp1_ns $C48 $B4 $D 1 $P $IL &
wait
cell pp2vp4n 0,1 2 c4_1024 dp1_ns $C48N $B4 $D 1 $P $IL &
cell pp4vp2c 2,3,4,5 4 c4_1024 dp1_ns $C48 $B4 $D 1 $P4 &
wait
cell pp4vp2n 0,1,2,3 4 c4_1024 dp1_ns $C48N $B4 $D 1 $P4
echo "# 2048 tokens per step, dp2"
( export NOSYNC_GA=1; cell d2_dp2_ns 0,1 2 c4_2048 - $C4 $B8 $D 2 )
cell d2_dp2 0,1 2 c4_2048 d2_dp2_ns $C4 $B8 $D 2 &
cell d2_ep2 2,3 2 c4_2048 d2_dp2_ns $C4 $B8 $D 2 $E 2 &
cell d2_pp2 4,5,6,7 4 c4_2048 d2_dp2_ns $C4 $B8 $D 2 $P &
wait
cell d2_vp2c 0,1,2,3 4 c4_2048 d2_dp2_ns $C4 $B8 $D 2 $P $IL &
cell d2_vp2n 4,5,6,7 4 c4_2048 d2_dp2_ns $C4N $B8 $D 2 $P $IL &
wait
echo; echo "# c4, 1024 tokens per step (4 x 256), reference dp1 with matched accumulation"
table dp1_ns dp1 dp1_rev pp2 vp2n vp2c pp2vp4n pp2vp4c pp4vp2n pp4vp2c
echo; echo "# c4, 2048 tokens per step (4 x 256 per rank), reference dp2 with matched accumulation"
table d2_dp2_ns d2_dp2 d2_ep2 d2_pp2 d2_vp2n d2_vp2c
echo RUN-PP-LOCAL-DONE
