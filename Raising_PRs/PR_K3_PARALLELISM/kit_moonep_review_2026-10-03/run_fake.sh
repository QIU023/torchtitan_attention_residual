#!/bin/bash
# The PR's GPU test against the fake MoonEP (logbook kit, 33327eb API) on the 5060: the copy only drops the multicast check.
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
W=${W:-$S/wt_moonep_clean}; R=$S/moonep_review_1003
FK=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_moonep_rewrite_2026-09-29/local/fake_moonep
. /workspace/venv_0928/bin/activate
cd $W
echo "tree $(git rev-parse --short HEAD) dirty=$(git status --short | wc -l)" > $R/fake_${TAG:-run}.log
PYTHONPATH=$FK:$W timeout 1800 python -m pytest $R/test_moonep_fake.py -rA -p no:cacheprovider ${K:+-k "$K"} >> $R/fake_${TAG:-run}.log 2>&1
echo "rc=$?" >> $R/fake_${TAG:-run}.log
