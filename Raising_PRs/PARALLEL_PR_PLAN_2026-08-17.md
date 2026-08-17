# Parallel PR plan, after PR-4025's 2026-08-17 state

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

## The problem with the four branches as they stand

`k3_pr_a_tp_kda`, `k3_pr_b_ep_grouped`, `k3_pr_c_pp_attnres`, `k3_pr_d_cp_ulysses` all point
at `a146d1bf2` and all carry the same 69-file diff. They are four names for one change, so
none of them is reviewable and they cannot be filed in parallel.

They also cannot be split cleanly along the axis their names suggest, because **all four
would touch `parallelize.py`** -- one file, four PRs, guaranteed conflict. Splitting by
feature without accounting for that is what produced four identical branches in the first
place.

## The split that actually parallelizes

One sequential prerequisite, then three genuinely independent PRs.

**PR 0 (prerequisite, not parallel): declarative model + parallelize skeleton.**
The config-driven conversion (eleven classes, adapters deleted, 172 Linear and 74 RMSNorm
now core's) plus a `parallelize_kimi_k3` whose body is FSDP-only -- i.e. functionally
equivalent to theirs -- with the per-axis application points as separate functions that do
nothing yet. Reviewable on its own: it changes no numerics (loss bit-identical to the
pre-conversion baseline on four cells) and it is the shape every other model in the repo
already has. Nothing else can land in parallel before this, because each later PR fills in
one of those functions.

**Then, in parallel, each touching one function and its own new file:**

| PR | adds | files it owns |
| --- | --- | --- |
| A: TP | `apply_tp_kimi_k3` + KDA's NoParallel/local-linear seam | `parallelize.py::apply_tp_*`, declarations in model |
| B: EP | `apply_ep_kimi_k3` + `verify_ep_applied` | `parallelize.py::apply_ep_*`, `moe.py` |
| C: CP | KCP (`kcp.py`) + Ulysses seams, per-mode wiring | `kcp.py`, `parallelize.py`'s CP block |
| D: PP | the AttnRes cross-stage adapter | `pipeline_adapter.py`, `layout.py` |

D is the largest and the least entangled -- it is two new files plus a `pipelining_fn`,
touching `parallelize.py` not at all. It can be filed first among the four, or last, without
affecting the others.

A and C interact but do not conflict: the `num_heads % (tp * cp) == 0` rule belongs to
whichever lands second, and the other's review does not need it.

## Sequencing constraints that are NOT about code

* **Do not file before PR-4025 merges.** Its eleven commits include a rename
  (`full_attention_layers` to 0-based) and a state-dict adapter refactor of 602 lines. Every
  one of our PRs references those names. Filing now means rebasing four PRs through someone
  else's review cycle. This was already the standing decision (`HANDOFF_2026-08-16`) and the
  new commits strengthen it.
* **PR 0 depends on where K3 lands.** If the maintainer keeps it in `models/`, PR 0 is a
  diff against their file. If it moves, PR 0 is a diff against a moved file. Cheap to redo,
  but not worth doing twice.
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
3. Build PR 0 as a branch off their merge commit, not off ours.
4. Reset the four `k3_pr_*` branches; they currently assert something false (four
   independent changes) and are worse than absent.
