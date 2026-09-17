source /workspace/kit/h200/env.sh; UV=/root/miniconda3/envs/py3.10/bin/uv; mkdir -p /workspace/results
$UV pip install "nvidia-cutlass-dsl[cu12]==4.6.0" > /workspace/uv_cutlass460.log 2>&1; echo "CUTLASS_RC=$? $(tail -1 /workspace/uv_cutlass460.log | cut -c1-120)"
$UV pip list 2>/dev/null | grep -i cutlass
python -c "import cutlass; print(\"cutlass.Vector:\", hasattr(cutlass, \"Vector\"))"
cd /workspace/moonep; for t in test_planning test_dispatch test_e2e; do echo "### $t"; timeout 900 torchrun --nproc_per_node=2 --master_port=41700 -m pytest -x -q tests/$t.py > /workspace/results/c460_$t.log 2>&1; echo "rc=$? $(grep -aE "passed|failed|error" /workspace/results/c460_$t.log | grep -v "^\[" | tail -1 | cut -c1-100)"; grep -a -m1 "Error\|error:" /workspace/results/c460_$t.log | cut -c1-160; done
cd $TITAN; timeout 900 torchrun --nproc_per_node=2 --master_port=41701 /workspace/kit/h200/moonep_hot_probe.py > /workspace/results/c460_hot_probe.log 2>&1; echo "hot rc=$?"; grep -a "HOT_PROBE_RESULT" /workspace/results/c460_hot_probe.log; grep -a -m1 "Error\|error:" /workspace/results/c460_hot_probe.log | cut -c1-160
echo "C460_CHECK_DONE"
