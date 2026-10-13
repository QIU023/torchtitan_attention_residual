#!/bin/bash
# Full GPU unit suite on one tree, main's pins and Elfie's torchao on the path, continuing past collection errors.
T=$1 OUT=$2
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
. /workspace/venv_1006/bin/activate
cd $T && PYTHONPATH=$S/main_pins:$T:$S/elfie_ao timeout 10800 python -m pytest -q -p no:cacheprovider -rfE --continue-on-collection-errors tests/unit_tests/gpu > $OUT 2>&1
echo "rc=$?" >> $OUT
