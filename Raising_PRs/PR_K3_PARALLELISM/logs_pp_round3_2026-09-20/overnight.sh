#!/bin/bash
# pp_review4 overnight chain (2026-09-20): sanity gate -> reference matrix -> K3 GPU unit tests -> validator under PP.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; TT=/tmp/wt_pp4312; O=$S/ppmx
TASKS=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/tasks
echo "start $(date -u +%FT%TZ)"
for i in $(seq 1 180); do grep -q "cell rc=" $TASKS/b5ipnr99r.output 2>/dev/null && break; sleep 5; done
grep "seed rc=\|cell rc=" $TASKS/b5ipnr99r.output
grep -q "cell rc=0 .*seed_loaded=1" $TASKS/b5ipnr99r.output || { echo "SANITY FAILED, stopping"; exit 1; }
echo "== matrix $(date -u +%T)"; bash $S/run_pp_local.sh > $O/run.log 2>&1; echo "matrix rc=$? $(date -u +%T)"; grep -E "rc=|^#" $O/run.log
echo "== K3 GPU unit tests $(date -u +%T)"
cd $TT && source /workspace/venv_bfx9/bin/activate && export PYTHONPATH=$TT
CUDA_VISIBLE_DEVICES=0 timeout 1800 python -m pytest tests/unit_tests/gpu/test_kimi_k3.py tests/unit_tests/gpu/test_kda_attention.py -q -p no:cacheprovider > $S/gpu_tests_pp_review4.log 2>&1; echo "gpu tests rc=$?"; tail -3 $S/gpu_tests_pp_review4.log
echo "== validator under PP $(date -u +%T)"
python3 - <<'PY'
p='torchtitan/models/kimi_k3/config_registry.py'; s=open(p).read()
if 'kimi_k3_debugmodel_c4_val' not in s:
    s += '''

def kimi_k3_debugmodel_c4_val() -> Trainer.Config:  # PROBE ONLY (not committed)
    from torchtitan.components.validate import Validator

    config = kimi_k3_debugmodel_c4()
    config.validator = Validator.Config(freq=1, steps=2, dataloader=replace(config.dataloader))
    return config
'''
    open(p,'w').write(s)
PY
B4="--training.num-tokens-per-train-step 1024 --training.num-tokens-per-microbatch-per-dp-rank 256"
P="--parallelism.pipeline_parallel_degree 2 --parallelism.num-pp-microbatches 4 --parallelism.pipeline_parallel_schedule Interleaved1F1B"
for nm in val_pp2vp2 val_dp1; do
  d=$O/$nm; rm -rf $d; mkdir -p $d; cp -r $O/seed_c4_1024/checkpoint $d/
  if [ $nm = val_pp2vp2 ]; then G=0,1; NP=2; X="$P"; else G=2; NP=1; X=""; fi
  CUDA_VISIBLE_DEVICES=$G TORCHINDUCTOR_CACHE_DIR=$O/ind_$nm TRITON_CACHE_DIR=$O/tri_$nm GN_FP32=1 timeout 1500 torchrun --nproc_per_node=$NP --master_port=$((30000+RANDOM%20000)) -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel_c4_val --debug.seed 42 --debug.deterministic --metrics.log_freq 1 $B4 --parallelism.data_parallel_shard_degree 1 $X --training.steps 3 --dump-folder $d > $O/$nm.log 2>&1
  echo "$nm rc=$? steps=$(grep -a -c 'step: ' $O/$nm.log) validation_lines=$(grep -a -c -i 'validat' $O/$nm.log) tracebacks=$(grep -a -c Traceback $O/$nm.log)"
  grep -a -i "validat" $O/$nm.log | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-160 | head -6
done
echo "OVERNIGHT-DONE $(date -u +%FT%TZ)"
