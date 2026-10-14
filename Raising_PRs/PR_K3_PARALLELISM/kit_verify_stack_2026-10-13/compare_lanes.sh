#!/bin/bash
# Compare the stack's and main's GPU lanes: counts per lane, then failures only on the stack, only on main, on both.
V=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad/verify_stack_1013
for l in lane1 lanem; do
  for t in stack main; do
    echo "$l $t: $(grep -E '^[0-9]+ (passed|failed)|^=+ .*(passed|failed|skipped).* in ' $V/${l}_$t.txt | tail -1) $(grep '^rc=' $V/${l}_$t.txt)"
    grep -E "^(FAILED|ERROR) " $V/${l}_$t.txt | sed -E 's/ - .*//' | sort > $V/${l}_${t}_fails.txt
  done
  echo "== $l failing only on the stack:"; comm -23 $V/${l}_stack_fails.txt $V/${l}_main_fails.txt
  echo "== $l failing only on main:"; comm -13 $V/${l}_stack_fails.txt $V/${l}_main_fails.txt
  echo "== $l failing on both: $(comm -12 $V/${l}_stack_fails.txt $V/${l}_main_fails.txt | wc -l)"
done
