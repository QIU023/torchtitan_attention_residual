#!/bin/bash
# pyrefly (pre-commit hook) on the amended opt-in head, on a copy
O=/workspace/h100_optin; L=$O/summary.txt
. /workspace/kit_setup/env_cu126.sh
cd /workspace/tt_new && git fetch -q --depth 4 origin cpmm_review1 && git checkout -q --detach FETCH_HEAD && echo "tt_new now $(git rev-parse HEAD)" >> $L
rm -rf /workspace/pyrefly_new2 && cp -r /workspace/tt_new /workspace/pyrefly_new2 && cd /workspace/pyrefly_new2
timeout 1800 pyrefly check --remove-unused-ignores --summarize-errors > $O/pyrefly_new2.log 2>&1
echo "pyrefly amended rc=$? $(grep -c '^ERROR' $O/pyrefly_new2.log) errors, hook changed $(git status --short | wc -l) files" >> $L
grep "^ERROR" $O/pyrefly_new2.log | sed "s/:[0-9]*:[0-9]*-[0-9]*:[0-9]*//; s/:[0-9]*:[0-9]*//" | sort > /tmp/e_new2.txt
grep "^ERROR" /workspace/h100_unit/pyrefly_tt_main.log | sed "s/:[0-9]*:[0-9]*-[0-9]*:[0-9]*//; s/:[0-9]*:[0-9]*//" | sort > /tmp/e_main.txt
diff -q /tmp/e_main.txt /tmp/e_new2.txt > /dev/null && echo "pyrefly amended: same error set as main" >> $L || echo "pyrefly amended: differs from main" >> $L
echo PYREFLY2_DONE >> $L
