# Shared by the run scripts. Set TT (branch tree), TT_PARENT (parent worktree), OUT.
TT=${TT:-$PWD}; TT_PARENT=${TT_PARENT:-$PWD/../tt_parent}; OUT=${OUT:-$PWD/tp_a100_out}; mkdir -p $OUT
CFG=${CFG:-kimi_k3_debugmodel}
D="--parallelism.data_parallel_shard_degree"; T="--parallelism.tensor_parallel_degree"
ST="--parallelism.spmd_backend spmd_types"; PD="--parallelism.spmd_backend partial_dtensor"; NOSP="--parallelism.no-enable-sequence-parallel"
COMMON="-m torchtitan.train --module kimi_k3 --config $CFG --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --checkpoint.enable --checkpoint.interval 100000"
seed() {  # one seed checkpoint per batch shape, built on the branch tree with 1 GPU
  local tag=$1; shift
  if [ ! -d $OUT/seed_$tag/checkpoint ]; then
    ( cd $TT && CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((30000+RANDOM%20000)) $COMMON --training.steps 1 --checkpoint.create_seed_checkpoint "$@" --dump-folder $OUT/seed_$tag > $OUT/seed_$tag.log 2>&1 )
  fi
}
cell() {  # cell <name> <tree> <gpus> <nproc> <seed tag> <steps> <flags...>
  local nm=$1 tree=$2 gpus=$3 np=$4 stag=$5 steps=$6; shift 6
  local d=$OUT/$nm; rm -rf $d; mkdir -p $d; cp -r $OUT/seed_$stag/checkpoint $d/
  ( cd $tree && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm torchrun --nproc_per_node=$np --master_port=$((30000+RANDOM%20000)) $COMMON --training.steps $steps "$@" --dump-folder $d > $OUT/$nm.log 2>&1 ); rm -rf $d/checkpoint
  echo "$nm rc=$? steps=$(grep -a -c 'step: ' $OUT/$nm.log)"
}
table() {  # table <reference name> <names...>
  python3 - "$OUT" "$@" <<'PY'
import re, sys, statistics
out, ref, names = sys.argv[1], sys.argv[2], sys.argv[3:]
def series(f):
    L, G = {}, {}
    for line in open(f, errors="ignore"):
        line = re.sub(r"\x1b\[[0-9;]*m", "", line)
        m = re.search(r"step: *(\d+)\s.*loss: *([0-9.-]+).*grad_norm: *([0-9.]+)", line)
        if m and float(m.group(2)) > 0: L[int(m.group(1))] = float(m.group(2)); G[int(m.group(1))] = float(m.group(3))
    return L, G
R = series(f"{out}/{ref}.log")
print(f"| cell | step 1 loss (diff) | step 10 loss (diff) | step 100 loss (diff) | mean abs loss diff, steps 1-100 | step 1 grad norm (diff) | step 10 (diff) | step 100 (diff) | steps |")
print("|---|---|---|---|---|---|---|---|---|")
for nm in [ref] + names:
    try: L, G = series(f"{out}/{nm}.log")
    except FileNotFoundError: print(f"| {nm} | missing |"); continue
    def fmt(X, Y, s):
        if s not in X: return "-"
        return f"`{X[s]:.6f}` ({abs(X[s]-Y[s])/Y[s]*100:.3g}%)" if nm != ref and s in Y else f"`{X[s]:.6f}`"
    mean = statistics.mean(abs(L[s]-R[0][s])/R[0][s] for s in L if s in R[0]) * 100 if nm != ref and L else 0.0
    print(f"| {nm} | " + " | ".join(fmt(L, R[0], s) for s in (1, 10, 100)) + f" | {mean:.3g}% | " + " | ".join(fmt(G, R[1], s) for s in (1, 10, 100)) + f" | {len(L)} |")
PY
}
