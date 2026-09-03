#!/bin/bash
# Waits on artifacts, never on process names: PP's DONE marker, then the CP script's existence.
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
until grep -q "PP MAIN30 MATRICES DONE" $SP/pp_main30_matrix.log 2>/dev/null; do sleep 60; done
until [ -x $SP/cp_main_matrix.sh ]; do sleep 30; done
$SP/cp_main_matrix.sh > $SP/cp_main_matrix.log 2>&1
$SP/qb_main_matrix.sh > $SP/qb_main_matrix.log 2>&1
echo "CHAIN DONE"
