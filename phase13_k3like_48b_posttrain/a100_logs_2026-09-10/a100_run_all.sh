#!/bin/bash
# The A100 x 8 program: numerics first, then the two 100-step TP/SP tables, then QB x EP for 100 steps.
until grep -q DEPS-DONE /workspace/setup_deps.log 2>/dev/null; do sleep 15; done
source /workspace/venv/bin/activate
export PYTHONPATH=/workspace/attn_gym_up
K=/workspace/matrix_scripts/tp_a100
export TT=/workspace/wt_tp TT_PARENT=/workspace/wt_parent OUT=/workspace/tp_a100_out
mkdir -p $OUT
stamp() { echo "=== $1 $(date +%T)"; }
stamp "tests"
( cd $TT && PYTHONPATH=/workspace/attn_gym_up:$TT timeout 1800 python -m pytest -q tests/unit_tests/cpu/test_kimi_k3_sp_splice.py tests/unit_tests/gpu/test_kda_attention.py tests/unit_tests/gpu/test_kimi_k3.py 2>&1 | tail -3 )
stamp "kernel sensitivity"
sed -i "s#/tmp/attn_gym_up#/workspace/attn_gym_up#" $K/kda_kernel_sensitivity.py
( cd $TT && CUDA_VISIBLE_DEVICES=0 timeout 900 python $K/kda_kernel_sensitivity.py 2>&1 | grep -v "W0\|Warning" | tail -12 )
stamp "fp32 loss"
( cd $K && timeout 3600 bash run_fp32_loss.sh 2>&1 | tail -8 )
stamp "fp32 grads"
( cd $K && timeout 3600 bash run_fp32_grads.sh 2>&1 | tail -12 )
stamp "bf16 100"
( cd $K && timeout 7200 bash run_bf16_100.sh 2>&1 | tail -22 )
stamp "fp32m 100"
( cd $K && timeout 7200 bash run_fp32m_100.sh 2>&1 | tail -22 )
stamp "qb ep 100"
cd /workspace/wt_qb && git apply --check /workspace/matrix_scripts/qb_probe_flavors.patch 2>&1 | tail -1 && git apply /workspace/matrix_scripts/qb_probe_flavors.patch && cp /workspace/matrix_scripts/load_probe.py torchtitan/components/load_probe.py && echo "probe flavors applied"
sed -e "s#MX=/workspace/torchtitan_attention_residual/phase13_k3like_48b_posttrain/matrix_scripts/mx3.sh#MX=/workspace/matrix_scripts/mx3.sh#" -e "s#VENV=/workspace/venv_bfx9#VENV=/workspace/venv#" -e "s#PYPRE=/tmp/attn_gym_up#PYPRE=/workspace/attn_gym_up#" -e "s#TITAN=/tmp/wt_qbrun4#TITAN=/workspace/wt_qb#g" /workspace/matrix_scripts/qb_dp8ep8.sh > /workspace/matrix_scripts/qb_dp8ep8_a100.sh
timeout 7200 bash /workspace/matrix_scripts/qb_dp8ep8_a100.sh 2>&1 | tail -6
stamp "done"
echo ALL-DONE
