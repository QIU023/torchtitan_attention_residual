#!/bin/bash
# PR 194033: the CI and lint failures with and without the onlyOn import line, on one environment.
# without = PR head 6e815dc26d's test/test_nn.py, with = review1 6245bda9bc's; everything else is the review1 tree.
R=/workspace/torchtitan_attention_residual/pytorch; PY=/workspace/venv_pr194033/bin/python
cd $R || exit 1
trap 'git -C $R checkout -q 6245bda9bc -- test/test_nn.py' EXIT
for state in without:6e815dc26d with:6245bda9bc; do
  name=${state%%:*}; rev=${state#*:}
  git show $rev:test/test_nn.py > test/test_nn.py
  echo "===== $name the line ($rev)"
  echo "--- lint, ruff 0.14.4 with pyproject.toml (CI's RUFF linter):"
  timeout 300 uvx --quiet ruff@0.14.4 check --config pyproject.toml test/test_nn.py 2>&1 | grep -E "F821|Undefined|All checks passed|Found [0-9]+ error" | head -3
  echo "--- collect all of test/test_nn.py (what every CI shard does first):"
  (cd /tmp && CUDA_VISIBLE_DEVICES=0 timeout 900 $PY -m pytest --collect-only -q $R/test/test_nn.py 2>&1 | grep -E "NameError|tests? collected|error" | tail -2)
  echo "--- the PR's test, CPU and CUDA:"
  CUDA_VISIBLE_DEVICES=0 timeout 900 $PY test/test_nn.py -k test_get_total_norm_dtype 2>&1 | grep -E "NameError|^Ran |^OK|FAILED" | tail -2
done
git checkout -q 6245bda9bc -- test/test_nn.py
echo "===== with the line, further checks"
echo "--- the PR's test under PYTORCH_TEST_WITH_DYNAMO=1 (the dynamo_wrapped shards):"
CUDA_VISIBLE_DEVICES=0 PYTORCH_TEST_WITH_DYNAMO=1 timeout 1200 $PY test/test_nn.py -k test_get_total_norm_dtype 2>&1 | grep -E "^Ran |^OK|FAILED|Error" | tail -2
echo "--- the existing clip_grad tests of test_nn next to it, CPU and CUDA:"
CUDA_VISIBLE_DEVICES=0 timeout 1800 $PY test/test_nn.py -k "clip_grad or total_norm" 2>&1 | grep -E "^Ran |^OK|FAILED|Error" | tail -2
echo "--- DTensor foreach_norm / foreach_powsum, 4 GPUs:"
CUDA_VISIBLE_DEVICES=0,1,2,3 timeout 1500 $PY test/distributed/tensor/test_math_ops.py -k foreach_norm -k foreach_powsum 2>&1 | grep -E "^Ran |^OK|FAILED" | tail -2
echo "--- tree after the run:"; git status --short | head -3; git log --oneline -1 | cat
