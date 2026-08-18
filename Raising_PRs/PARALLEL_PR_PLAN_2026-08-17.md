# Parallel PR plan: the four parallelism PRs to file the moment PR-4025 merges

## Where these sit relative to PR-4025

The RFC covering all of K3 is ours. PR-4025 took the eager half -- module-by-module port with
class substitutions, FSDP2 only -- and that division is not a loss: the distributed training
half, which is the hard part, is still ours to land.

How completely theirs stops at eager is worth quoting, because it defines our four PRs
exactly. `KimiK3Model.Config.update_from_config` in `c3a80b01e`:

    unsupported = {
        "tensor parallel": parallelism.tensor_parallel_degree,
        "pipeline parallel": parallelism.pipeline_parallel_degree,
        "context parallel": parallelism.context_parallel_degree,
        "expert parallel": parallelism.expert_parallel_degree,
    }
    enabled = [name for name, degree in unsupported.items() if degree > 1]
    if enabled:
        raise NotImplementedError(
            "Kimi K3 supports FSDP2 data parallelism only; "
            f"disable {', '.join(enabled)}."
        )

Four axes, four raises. Two more of the same kind: `parallelize.py` hardcodes `ep_degree=1`
into the FSDP call, and the MoE uses `LocalTokenDispatcher` (rank-local, no all-to-all). They
also refuse `packing_buffer_size > 0`.

**So each of our PRs deletes one of those four lines and supplies the implementation behind
it.** That is a much better PR story than a refactor of someone else's model: nothing to
reconcile, one `NotImplementedError` removed per PR, and a matrix behind each.

**File them immediately after 4025 merges, PP and CP first.** They are the two that are hard,
the two nobody else has, and the two with the most evidence behind them here -- the
cross-stage adapter validated to PP8xVP4, and KCP verified forward and backward at cp=2/4/8
against a single-rank reference. TP and EP are more conventional and can follow.

## What changed upstream

PR-4025 moved `1da44c16a -> c3a80b01e`: eleven of its own commits plus a merge of
upstream/main. Its own eleven, in order of what they mean for us:

| commit | why it matters here |
| --- | --- |
| support full kimi k3 and refactor readme | it is no longer debug-model only |
| change full_attention_layers to starting with 0 | **index base flipped**; ours is 1-based |
| remove parallelism config | it dropped config fields we may have assumed |
| reuse VisionAttention | vision attention now comes from `models/common` |
| refactor KimiK3StateDictAdapter (602 lines changed) | our adapter diverges further |
| refactor KimiGroupedExperts to use self.grouped_mm | touches the seam our EP work uses |
| support flex attention | a path we do not have |
| fix pyrefly / refactor test case / snapshot_download pin / fix NIT | mechanical |

**The structural fact that decides the plan is unchanged and now measurable more sharply:**
their `parallelize.py` is 99 lines with ZERO `ColwiseParallel`/`RowwiseParallel`/
`parallelize_module`, and it calls exactly `apply_fsdp_to_decoder` and
`apply_fsdp_to_vision_encoder`. Their `model.py` has ZERO `sharding_config` occurrences and
there is no `sharding.py`. So they have FSDP2 and nothing else -- not even the declarative
sharding that qwen3_5 carries.

Ours is 1656 lines because it holds TP, EP, CP and PP. **That means our parallelism work is
purely additive to their tree, which is good for splitting it up -- there is nothing to
reconcile, only things to add.**

## What the four branches were

All four pointed at `a146d1bf2` with the same 69-file diff -- four names for one change, so
none was reviewable. Reset now; see the next section.

## The split, as four diffs with no overlapping file

Done. `Raising_PRs/diffs/` holds one diff per axis, and the four branches now point at one
axis each instead of four names for one 69-file change:

| branch | files | diff lines |
| --- | --- | --- |
| `k3_pr_a_tp_kda` | `parallelize.py` | 1754 |
| `k3_pr_b_ep_grouped` | `moe.py`, `quantile_balance.py` | 514 |
| `k3_pr_c_cp_kcp` | `kcp.py`, `vit_cp_plan.py` | 444 |
| `k3_pr_d_pp_attnres` | `pipeline_adapter.py`, `layout.py` | 2066 |

Six of those seven files do not exist upstream at all, which is what makes the split clean:
`moe.py`, `quantile_balance.py`, `kcp.py`, `vit_cp_plan.py`, `pipeline_adapter.py`,
`layout.py` are ours. Only `parallelize.py` is shared, and it belongs to exactly one of the
four (A), so no two PRs touch the same file. The earlier worry about all four colliding in
`parallelize.py` was wrong: TP is the axis that lives there, and the other three axes live in
their own modules and are reached from it.

An earlier draft of this plan proposed a "PR 0" holding the declarative model conversion plus
an FSDP-only skeleton. That was empty as written -- the skeleton is what upstream already
has, and the conversion largely tracks work they did themselves. What genuinely sits outside
the four diffs is the AttnRes model (`attn_res.py`, `attn_res_model.py`), which upstream
references by config in seven places but does not implement. That is a fifth diff if it is
ever needed separately, not a prerequisite dressed up as one.

**What the four diffs are and are not.** Each is a reviewable slice of one axis. None is a
standalone build: they need the AttnRes model and the converted model definition, which are
outside their file sets. Each branch's commit message says so rather than implying the branch
compiles.

## Sequencing constraints that are NOT about code

* **Do not file before PR-4025 merges.** Its eleven commits include a rename
  (`full_attention_layers` to 0-based) and a state-dict adapter refactor of 602 lines. Every
  one of our PRs references those names. Filing now means rebasing four PRs through someone
  else's review cycle. This was already the standing decision (`HANDOFF_2026-08-16`) and the
  new commits strengthen it.
* **Where K3 lands changes only diff A.** If the maintainer moves `kimi_k3` out of
  `models/`, `parallelize.py`'s path moves with it and diff A has to be regenerated. The
  other three add new files and are path-independent.
* **The 0-based index flip is a real correctness item for us**, not a rename: our
  `kda_layers` and `full_attn_layers` are 1-based, and `state_dict_adapter` already had a
  bug from exactly this confusion (`kda_layers_zero_based` used where `is_kda_layer` was
  meant, disagreeing on layer 0). Align that BEFORE splitting anything, or the split
  propagates the ambiguity into four PRs.

## What to do next, in order

1. Re-vendor the reference tree at `c3a80b01e` (currently pinned at `0cadf15e0`) and re-run
   `DIFF_VS_4025` against it. The three-bucket breakdown in that document was computed
   against the old base and the state-dict adapter bucket in particular is now stale.
2. Resolve the 1-based/0-based layer index against their new convention.
3. Regenerate the four diffs off their merge commit once it lands, not off `upstream/main`
   as they are now.
4. Decide whether the AttnRes model goes as a fifth diff or rides with D -- D is the axis
   that cannot be reviewed without it.
