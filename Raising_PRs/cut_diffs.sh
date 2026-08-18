#!/usr/bin/env bash
# Re-cut the five per-axis diffs in diffs/ from the current torchtitan tree.
#
#   bash Raising_PRs/cut_diffs.sh [BASE]        # BASE defaults to upstream/main
#
# The diffs under diffs/ were originally cut by hand, with no record of the
# groupings or the base. A hand-cut diff goes stale the first time the tree moves
# and there is then no way to tell whether it is stale or whether the split
# changed -- which is what happened when apply_cp_kimi_k3 was extracted a few
# hours after the cut.
#
# Groupings are the ones in REUSE_AUDIT_2026-08-18.md. The point of the split is
# that the four axis PRs touch disjoint file sets; 0_shared is what they cannot
# avoid sharing.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../torchtitan" && pwd)
BASE=${1:-upstream/main}
OUT="$HERE/diffs"
M=torchtitan/models/kimi_k3

mkdir -p "$OUT"
cd "$REPO"

cut() {
    local name=$1; shift
    git diff "$BASE" -- "$@" > "$OUT/$name.diff"
    printf '%-24s %7s lines  %s files\n' "$name" \
        "$(wc -l < "$OUT/$name.diff")" "$(grep -c '^diff --git' "$OUT/$name.diff" || true)"
}

cut 0_shared_parallelize $M/parallelize.py $M/knobs.py
cut a_tp                 $M/quant_scope.py
cut b_ep_moe             $M/moe.py $M/quantile_balance.py
cut c_cp_kcp_dyncp       $M/kcp.py $M/vit_cp_plan.py $M/multimodal_model.py $M/moonvit.py
cut d_pp_attnres_dep     $M/pipeline_adapter.py $M/layout.py $M/attn_res.py \
                         $M/attn_res_model.py $M/dep_bubble_plan.py \
                         $M/dep_bubble_runtime.py $M/dep_bubble_backward.py \
                         $M/vit_prefetch.py

echo "base: $BASE ($(git rev-parse --short "$BASE"))"
echo "tree: $(git rev-parse --short HEAD)"

# These five are the PARALLELISM split, not the whole model folder. Say so out loud,
# because 17 of 86 changed files is exactly the kind of number that gets read as
# "everything" once the header scrolls off.
total=$(git diff "$BASE" --name-only -- "$M/" | wc -l)
covered=$(grep -ho '^+++ b/.*' "$OUT"/*.diff | sort -u | wc -l)
echo "covers $covered of $total changed files under $M/"
echo "the rest is the model itself (model.py, config_registry.py, state_dict_adapter.py)"
echo "plus its tests -- that is PR-4025's territory, not the axis PRs'"
