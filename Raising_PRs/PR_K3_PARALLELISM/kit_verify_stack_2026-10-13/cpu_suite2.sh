#!/bin/bash
# Full CPU unit suite on one tree, GPUs hidden: pass 1 with Elfie's source torchao (MX QAT needs it), continuing past
# collection errors; pass 2 runs the files whose collection needs a versioned torchao with the venv's nightly instead.
T=$1 OUT=$2
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
. /workspace/venv_1006/bin/activate
cd $T
CUDA_VISIBLE_DEVICES= PYTHONPATH=$S/main_pins:$T:$S/elfie_ao timeout 3600 python -m pytest -q -p no:cacheprovider -rfE --continue-on-collection-errors tests/unit_tests/cpu > $OUT.pass1 2>&1
echo "rc=$?" >> $OUT.pass1
FILES="tests/unit_tests/cpu/components/test_validator.py tests/unit_tests/cpu/models/flux/test_dataset_flux.py tests/unit_tests/cpu/models/flux/test_flux_inference.py tests/unit_tests/cpu/models/flux/test_flux_sharding.py tests/unit_tests/cpu/recipes/test_debug_config_defaults.py tests/unit_tests/cpu/test_integration_test_definitions.py"
CUDA_VISIBLE_DEVICES= PYTHONPATH=$S/main_pins:$T timeout 1800 python -m pytest -q -p no:cacheprovider -rfE $FILES > $OUT.pass2 2>&1
echo "rc=$?" >> $OUT.pass2
