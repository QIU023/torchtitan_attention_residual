#!/bin/bash
# pre-commit's pyrefly hook as configured (pyrefly 0.45.1 check --remove-unused-ignores --summarize-errors), on copies of the
# PR head and main, so the hook's ignore removal shows up as a diff instead of touching the trees
O=/workspace/h100_unit; L=$O/summary.txt
. /workspace/kit_setup/env_cu126.sh
for t in tt_cpmm tt_main; do
  rm -rf /workspace/pyrefly_$t && cp -r /workspace/$t /workspace/pyrefly_$t && cd /workspace/pyrefly_$t
  timeout 1800 pyrefly check --remove-unused-ignores --summarize-errors > $O/pyrefly_$t.log 2>&1
  echo "pyrefly $t rc=$? $(grep -c '^ERROR' $O/pyrefly_$t.log) errors, files changed by the hook: $(git status --short | wc -l)" >> $L
  git diff > $O/pyrefly_$t.diff
done
echo PYREFLY_DONE >> $L
