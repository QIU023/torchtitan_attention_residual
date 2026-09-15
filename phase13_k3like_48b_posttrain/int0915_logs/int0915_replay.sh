#!/bin/bash
# Replay commits onto /tmp/wt_int0915 until a conflict or the QB marker. Usage: int0915_replay.sh <commit>...
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad; MAP=$S/int0915_map.txt
cd /tmp/wt_int0915
while [ $# -gt 0 ]; do
  c=$1
  if [ "$c" = "QB_4577" ]; then echo "STOP at QB_4577"; echo "remaining: $*"; exit 3; fi
  out=$(git cherry-pick $c 2>&1); rc=$?
  if [ $rc -ne 0 ]; then
    if echo "$out" | grep -q "is now empty\|nothing to commit"; then git cherry-pick --skip >/dev/null 2>&1; echo "$c -> (empty, skipped)" >> $MAP; shift; continue; fi
    u=$(git diff --name-only --diff-filter=U | tr '\n' ' ')
    if [ -z "$u" ] && echo "$out" | grep -q "using previous resolution"; then
      git add -u && GIT_EDITOR=true git cherry-pick --continue >/dev/null 2>&1 && { echo "$c -> $(git rev-parse --short HEAD) (rerere)" >> $MAP; shift; continue; }
    fi
    echo "STOP at $c: conflicts [$u]"; echo "$out" | grep -i "previous resolution\|CONFLICT" | head -6; echo "remaining: $*"; exit 1
  fi
  echo "$c -> $(git rev-parse --short HEAD)" >> $MAP; shift
done
echo "ALL DONE"
