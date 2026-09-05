#!/bin/bash
# rerun after the singleton-axis guard: A'' 482f21753, B'' daa927084, CP'' 88f232df2 (lint pass); QB unchanged (a4658eefe)
SP=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
$SP/decl_matrix.sh > $SP/decl_matrix.log 2>&1
$SP/tpsp2_matrix.sh > $SP/tpsp2_matrix.log 2>&1
$SP/cp6_matrix.sh > $SP/cp6_matrix.log 2>&1
$SP/qb3b_matrix.sh > $SP/qb3b_matrix.log 2>&1
echo "CHAIN2 DONE"
