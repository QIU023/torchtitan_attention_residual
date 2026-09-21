#!/bin/bash
# bf16, reference vs pp2 x vp2 cache off, 2 steps: the clip coefficient differs by one unit at step 2 (measured), so dump
# the gradients after clipping and the master parameters after the step-2 update; where does the unit vanish?
set -eu
TT=/workspace/tt_pp; OUT=/workspace/results/pp_r3_bf16clip; SRC=/workspace/results/pp_r3; LOG=/workspace/results/pp_r3_run.log; mkdir -p $OUT
export VIRTUAL_ENV=/workspace/venv_k3 PATH=/workspace/venv_k3/bin:$PATH PYTHONPATH=$TT NCCL_NVLS_ENABLE=0 GN_FP32=1 GN_REPR=1 PARAM_DUMP_STEP=2
cd $TT && python3 - <<'PY'
import pathlib
p = pathlib.Path("torchtitan/training_engine.py"); s = p.read_text()
if "GRADCLIP_DUMP" not in s:
    old = '''        if __import__("os").environ.get("GN_REPR") == "1":  # LOCAL PROBE HACK (not committed)
'''
    new = '''        if __import__("os").environ.get("GRADCLIP_DUMP") and current_step == int(__import__("os").environ.get("PARAM_DUMP_STEP", "1")):  # LOCAL PROBE HACK (not committed): gradients after the clip
            _d = {}
            for _m in self.model_parts:
                for _n, _p in _m.named_parameters():
                    if _p.grad is None:
                        continue
                    _g = _p.grad
                    _g = _g.full_tensor() if hasattr(_g, "full_tensor") else _g
                    _d[_n] = _g.detach().cpu()
            _dd, _bb = __import__("os").path.split(__import__("os").environ["GRADCLIP_DUMP"])
            torch.save(_d, __import__("os").path.join(_dd, f"gradsclipped_{_bb}.rank{torch.distributed.get_rank()}.pt"))
''' + old
    assert s.count(old) == 1; p.write_text(s.replace(old, new)); print("GRADCLIP_DUMP added")
PY
python3 -m py_compile torchtitan/training_engine.py
COMMON="-m torchtitan.train --module kimi_k3 --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 2"
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256 --parallelism.data_parallel_shard_degree 1"
IL="--parallelism.pipeline_parallel_schedule Interleaved1F1B"; P2="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4"
cell() {
  local nm=$1 gpus=$2 np=$3 port=$4 cfg=$5; shift 5
  local d=$OUT/$nm; rm -rf $d $OUT/ind_$nm $OUT/tri_$nm; mkdir -p $d; cp -r $SRC/seed_c4_1024/checkpoint $d/
  cp -r $SRC/ind_dp1_ns_seqsum $OUT/ind_$nm; cp -r $SRC/tri_dp1 $OUT/tri_$nm
  set +e
  ( cd $TT && CUDA_VISIBLE_DEVICES=$gpus TORCHINDUCTOR_CACHE_DIR=$OUT/ind_$nm TRITON_CACHE_DIR=$OUT/tri_$nm PARAM_DUMP=$OUT/$nm GRADCLIP_DUMP=$OUT/$nm \
      torchrun --nproc_per_node=$np --master_port=$port $COMMON --config $cfg "$@" --dump-folder $d > $OUT/$nm.log 2>&1 )
  local rc=$?; set -e; rm -rf $d/checkpoint
  echo "$nm rc=$rc steps=$(grep -a -c 'step: ' $OUT/$nm.log) dumps=$(ls $OUT/*_$nm.rank*.pt 2>/dev/null | wc -l) $(date -u +%T)" >> $LOG
}
echo "== BC: bf16 step 2, gradients after the clip and the parameters after the update $(date -u +%T)" >> $LOG
( export NOSYNC_GA=1; cell ref 0 1 47601 kimi_k3_debugmodel_c4 $B4 ) &
cell vp2n 2,3 2 47602 kimi_k3_debugmodel_c4_pp_naive $B4 $P2 $IL &
wait
cd $OUT && python3 - >> $OUT/compare.txt 2>&1 <<'PY'
import torch, glob
def load(kind, nm):
    d = {}
    for f in sorted(glob.glob(f"{kind}_{nm}.rank*.pt")): d.update(torch.load(f))
    return d
for kind in ("gradsclipped", "params"):
    a, b = load(kind, "ref"), load(kind, "vp2n"); common = sorted(set(a) & set(b))
    dt = de = tot = 0; mx = 0
    for n in common:
        x, y = a[n].float(), b[n].float(); tot += x.numel(); ne = x != y
        if ne.any():
            dt += 1; de += int(ne.sum()); mx = max(mx, int((x.view(torch.int32) - y.view(torch.int32)).abs()[ne].max()))
    bf = sum(1 for n in common if a[n].float().to(torch.bfloat16).ne(b[n].float().to(torch.bfloat16)).any()) if kind == "gradsclipped" else -1
    print(f"{kind} at step 2: {dt}/{len(common)} tensors differ, {de}/{tot} elements, max {mx} fp32 units" + (f"; grads that are exactly bf16 values: {sum(1 for n in common if torch.equal(a[n].float().to(torch.bfloat16).float(), a[n].float()))}/{len(common)}" if kind == "gradsclipped" else ""))
PY
cat $OUT/compare.txt >> $LOG; echo "BC-DONE $(date -u +%T)" >> $LOG
