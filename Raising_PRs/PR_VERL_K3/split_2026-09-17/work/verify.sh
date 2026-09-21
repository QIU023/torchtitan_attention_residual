#!/bin/bash
# Apply the seven patches in order on a scratch copy of BASE; per stage: git apply --check, ast on the
# changed .py files, ruff F821/F811/F822 (/venv/main's ruff), and the cumulative test files the stages
# introduce (venv_verl, torchtitan = $TITAN, patch 00's vLLM version shim overlaid for the tests only).
set -uo pipefail
W=$(cd "$(dirname "$0")" && pwd); K=$(dirname "$W")
REPO=$(grep -oP 'REPO = "\K[^"]+' $W/splitlib.py); BASE=$(grep -oP 'BASE = "\K[^"]+' $W/splitlib.py); HEAD=$(grep -oP 'HEAD = "\K[^"]+' $W/splitlib.py)
TITAN=${TITAN:-/tmp/wt_int0922b}
T=$(mktemp -d /tmp/verl_split_verify.XXXX); git -C $REPO archive $BASE | tar -x -C $T; git -C $T init -q && git -C $T add -A && git -C $T -c user.email=x@y -c user.name=x commit -q -m base
git -C $REPO show $HEAD:verl/third_party/vllm/__init__.py > $T.shim
echo "base $BASE head $HEAD scratch $T torchtitan $TITAN"
tests=()
for p in 01_engine_tp_packed_and_compat 02_engine_pipeline 03_engine_ep_lora_qat_sync 04_engine_context_parallel 05_kimi_k3 06_metrics_logprob_diff local_env_and_diagnostics; do
  f=$K/$p.patch
  if git -C $T apply --check $f 2>$T.err; then echo "apply --check OK: $p.patch"; else echo "apply --check FAIL: $p.patch: $(head -n 3 $T.err)"; exit 1; fi
  git -C $T apply $f
  for t in $(grep '^+++ b/tests/' $f | sed 's|^+++ b/||' | grep '\.py$'); do [[ " ${tests[*]} " == *" $t "* ]] || tests+=($t); done
  changed=$(git -C $T diff --name-only | grep '\.py$'); n=$(echo "$changed" | grep -c .)
  ast=$(cd $T && for c in $changed; do /venv/main/bin/python -c "import ast,sys; ast.parse(open('$c').read())" 2>/dev/null || echo "$c"; done | wc -l)
  ruff=$(cd $T && /venv/main/bin/ruff check --select F821,F811,F822 $changed 2>&1 | tail -n 1)
  echo "  stage ${p:0:2}: $n changed py files, ast errors $ast, ruff $ruff"
  git -C $T add -A && git -C $T -c user.email=x@y -c user.name=x commit -q -m "$p"
  if [ ${#tests[@]} -gt 0 ] && [ "$p" != local_env_and_diagnostics ]; then
    cp $T.shim $T/verl/third_party/vllm/__init__.py
    r=$(cd $T && source /workspace/venv_verl/bin/activate && VERL_VLLM_VERSION=0.18.0 PYTHONPATH=$T:$TITAN:/workspace/pylib/attn_gym_main timeout 900 python -m pytest -q -p no:cacheprovider "${tests[@]}" 2>&1 | grep -E '^FAILED|^ERROR|passed|failed|error' | tail -n 12)
    echo "  tests (${#tests[@]} files): $r" | sed '2,$s/^/    /' 
    git -C $T checkout -q -- verl/third_party/vllm/__init__.py
  fi
done
# build.py asserts final tree == HEAD tree with git plumbing (build.log); the scratch copy skips ignored-but-tracked files, so no tree compare here.
echo "changed files at the last stage: $(git -C $T diff --name-only $(git -C $T rev-list --max-parents=0 HEAD) HEAD | wc -l) (branch: $(git -C $REPO diff --name-only $BASE $HEAD | wc -l))"
rm -rf $T $T.shim $T.err
