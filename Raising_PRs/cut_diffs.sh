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
BASE=${1:-$(cd "$HERE/../torchtitan" && git merge-base upstream/main HEAD)}
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

# The base: the model, plus the parallelize entry with TP/CP/EP raising. Every axis PR
# sits on this. multimodal_model.py and moonvit.py belong HERE, not in the CP group where
# they used to sit -- CP uses their dynamic-CP execution half, but PP's DEP path imports
# KimiK3MultimodalModel and KimiK3ViTStage from them too, so assigning them to CP made PP
# depend on the CP PR for no reason. They are model files.
cut 0_base               $M/parallelize.py $M/knobs.py $M/multimodal_model.py \
                         $M/moonvit.py torchtitan/models/__init__.py
cut a_tp                 $M/quant_scope.py
cut b_ep_moe             $M/moe.py $M/quantile_balance.py torchtitan/models/utils.py
cut c_cp_kcp_dyncp       $M/kcp.py $M/vit_cp_plan.py
cut d_pp_attnres_dep     $M/pipeline_adapter.py $M/layout.py $M/attn_res.py \
                         $M/attn_res_model.py $M/dep_bubble_plan.py \
                         $M/dep_bubble_runtime.py $M/dep_bubble_backward.py \
                         $M/vit_prefetch.py \
                         torchtitan/components/lr_scheduler.py \
                         torchtitan/components/optimizer.py \
                         torchtitan/distributed/fsdp.py

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

# Coverage assertion. The five diffs once silently omitted every modified upstream core
# file -- PR23's slice was missing the lr_scheduler and optimizer changes that let a
# LoRA+PP stage holding only frozen weights exist at all, and the fsdp.py change whose
# absence is a deadlock rather than an error. Nothing caught it because nothing checked.
#
# Files owned by a separately-filed PR are declared here, not silently tolerated.
declare -A ELSEWHERE=(
  [torchtitan/distributed/utils.py]="PR26, filed upstream as 4135"
  [torchtitan/models/common/moe.py]="PR27 (gate_up_combine) + PR28 (router_input_BLD)"
  [torchtitan/models/common/moe_sharding.py]="PR28"
)
missing=0
while IFS= read -r f; do
    case "$f" in "$M"/*) continue ;; esac          # model folder: handled by the note above
    grep -q "^+++ b/$f\$" "$OUT"/*.diff && continue
    if [ -n "${ELSEWHERE[$f]:-}" ]; then
        echo "  not in the five, by design: $f  -> ${ELSEWHERE[$f]}"
    else
        echo "  UNASSIGNED: $f"; missing=1
    fi
done < <(git diff "$BASE" --name-only --diff-filter=M)
[ "$missing" -eq 0 ] || { echo "a modified upstream file belongs to no PR; assign it or declare it"; exit 1; }
