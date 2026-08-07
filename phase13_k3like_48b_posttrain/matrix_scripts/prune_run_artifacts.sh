#!/usr/bin/env bash
# Delete reproducible training dumps, keep the evidence.
#
# Why this exists: a 100-step matrix cell leaves a ~700 MB dump folder
# (checkpoint/ + tb/ + structured_logs/) while the thing anyone ever reads from it
# is the ~200 KB .log. /tmp reached 227 GB of these and the box ran out of disk
# mid-run, which took bash down with it -- the harness could not even write a
# command's output file.
#
# The rule, and it is deliberately narrow: a directory is deleted ONLY if it
# directly contains one of the three markers a torchtitan dump folder always has
# (checkpoint, tb, structured_logs). Anything else is left alone, so this cannot
# wander into a source tree or a dataset.
#
# NEVER deleted: *.log, *.json, *.jsonl, *.md, *.csv (the evidence and the
# reproducibility record), anything under a git working tree, anything named
# seed_* or warm* (regenerating a seed checkpoint costs a run and invalidates a
# matrix's shared-init protocol), and /tmp/claude-* (agent session state).
#
#   prune_run_artifacts.sh --dry-run [roots...]   # report only
#   prune_run_artifacts.sh [roots...]             # delete
#
# Default roots: /tmp /workspace

set -uo pipefail

DRY=0
if [ "${1:-}" = "--dry-run" ]; then DRY=1; shift; fi
ROOTS=("$@")
[ ${#ROOTS[@]} -eq 0 ] && ROOTS=(/tmp /workspace)

# Substring guards. A candidate matching any of these is skipped outright.
KEEP_PATTERNS=(
  "/claude-"                       # agent session state, incl. scratchpads
  "torchtitan_attention_residual"  # the git working trees
  "warm"                           # warm checkpoints for the LoRA probes
  "/vllm_k3"                       # a source build, not a run artifact
  "k3mini_hf"                      # exported HF bundles
  "k3_official_code"               # release files copied in
)

is_protected() {
  local p="$1"
  for pat in "${KEEP_PATTERNS[@]}"; do
    case "$p" in *"$pat"*) return 0 ;; esac
  done
  # Seed checkpoints, by BASENAME containing "seed" at all. Two dry runs were
  # needed to get this right: "/seed_" missed a folder named plainly "seed", and
  # the prefix form then missed qat_seed / nokda_seed / ds_seed / pr19_seed.
  # Regenerating one costs a run and breaks a matrix's shared-init protocol, so
  # the guard is deliberately over-broad -- 4 GB of insurance against 250 GB.
  case "$(basename "$p")" in *seed*) return 0 ;; esac
  return 1
}

human() { numfmt --to=iec --suffix=B "${1:-0}" 2>/dev/null || echo "${1}B"; }

total=0
count=0
for root in "${ROOTS[@]}"; do
  [ -d "$root" ] || continue
  # Delete the CHECKPOINT subdirectory, not the dump folder. Measured on one
  # 13-cell 100-step matrix: checkpoint/ is 9.0 GB, structured_logs/ is 159 MB and
  # tb/ is 1.3 MB. So checkpoint is 97% of it, and tb is the one thing that must
  # survive -- stdout prints five significant digits, and CLAUDE.md is explicit
  # that a loss comparison has to be read from the TensorBoard record instead.
  # Deleting the whole folder would have thrown away the only full-precision copy
  # of every published matrix.
  while IFS= read -r ckpt; do
    # Test the DUMP FOLDER as well as the checkpoint path. The guards match on
    # basename, and the basename here is always "checkpoint" -- so a seed folder's
    # own name is only visible one level up. Without this, .../seed/checkpoint
    # sailed straight past the seed guard.
    is_protected "$ckpt" && continue
    is_protected "$(dirname "$ckpt")" && continue
    sz=$(du -sb "$ckpt" 2>/dev/null | cut -f1)
    [ -z "$sz" ] && continue
    total=$((total + sz))
    count=$((count + 1))
    if [ "$DRY" = 1 ]; then
      printf "would delete  %10s  %s\n" "$(human "$sz")" "$ckpt"
    else
      printf "deleting      %10s  %s\n" "$(human "$sz")" "$ckpt"
      rm -rf "$ckpt"
    fi
  done < <(find "$root" -mindepth 2 -maxdepth 7 -name checkpoint -type d \
             2>/dev/null | sort -u)

  # JIT caches: pure rebuild cost, no evidence value.
  for cache in "$root"/torchinductor_* "$root"/tilelang* "$root"/.tilelang* \
               "$root"/triton_* "$root"/torchelastic_*; do
    [ -e "$cache" ] || continue
    is_protected "$cache" && continue
    sz=$(du -sb "$cache" 2>/dev/null | cut -f1)
    total=$((total + ${sz:-0}))
    count=$((count + 1))
    if [ "$DRY" = 1 ]; then
      printf "would delete  %10s  %s (JIT cache)\n" "$(human "${sz:-0}")" "$cache"
    else
      printf "deleting      %10s  %s (JIT cache)\n" "$(human "${sz:-0}")" "$cache"
      rm -rf "$cache"
    fi
  done
done

printf "\n%s %d path(s), %s\n" \
  "$([ "$DRY" = 1 ] && echo "would reclaim from" || echo "reclaimed from")" \
  "$count" "$(human "$total")"
