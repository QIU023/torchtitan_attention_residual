#!/bin/bash
# PR 4499 round 3 on 4 x H100: kit tests first, then run_v3.sh. Logs under /workspace/tp_h100_out.
source /workspace/venv/bin/activate
export TT=/workspace/tt TT_PARENT=/workspace/tt_parent OUT=/workspace/tp_h100_out; mkdir -p $OUT
cd $TT
{ echo "branch $(git log --oneline -1 | cut -c1-9) parent $(git -C $TT_PARENT log --oneline -1 | cut -c1-9)"
  python -c "import torch, importlib.metadata as m; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'attn-gym', m.version('attn-gym'), 'spmd_types', m.version('spmd_types'))"
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1; } > $OUT/versions.txt 2>&1
python -m pytest -q tests/unit_tests/cpu/test_integration_test_definitions.py tests/unit_tests/cpu/test_no_new_cli_options.py > $OUT/tests_cpu.log 2>&1; echo "cpu rc=$?" > $OUT/tests_rc.txt
python -m pytest -q tests/unit_tests/gpu/test_kimi_k3.py tests/unit_tests/gpu/test_kda_attention.py > $OUT/tests_gpu.log 2>&1; echo "gpu rc=$?" >> $OUT/tests_rc.txt
touch $OUT/TESTS-DONE
bash /workspace/kit_tp/run_v3.sh > $OUT/run.log 2>&1
touch $OUT/MATRIX-DONE
