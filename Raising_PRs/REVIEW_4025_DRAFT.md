# Review draft for pytorch/torchtitan#4025 (Kimi K3 eager reference)

Status: DRAFT, not posted. Tone check before sending -- we want the interfaces,
not credit. Everything below is either a question or an offer.

---

## Comment 1 (top-level): what we have, and what we would like to line up with

Thanks for landing this -- an eager reference is the right first step, and
having `models/kimi_k3/` exist upstream unblocks a lot.

We have been building the parallelism side of K3 in a fork since the model's
release, and we would like to converge rather than diverge. The short version:
this PR explicitly scopes out TP, CP, PP and AC; we have all four working
against the K3 architecture (KDA + Gated MLA + LatentMoE + Block AttnRes,
including the multimodal tower), and we would rather build them on your module
layout than carry a parallel one.

Concretely, our fork currently passes a 13-leg text matrix (dp1, fsdp2, pp2,
cp2, tp2, and every 3-of-4 combination with EP on and off) and a 12-leg
multimodal matrix (FSDP/TP/CP/PP/EP combinations over MoonViT-V2 + the K3
backbone), 10 steps each, seed-42 deterministic. Branches, each ready to rebase
onto this PR's layout:

| | scope | branch |
|---|---|---|
| PR-A | TP: KDA NoParallel + Gated MLA + LatentMoE placements | `k3_pr_a_tp_kda` |
| PR-B | EP + grouped-GEMM | `k3_pr_b_ep_grouped` |
| PR-C | PP + Block AttnRes (blocked on ask 1 below) | `k3_pr_c_pp_attnres` |
| PR-D | CP: Ulysses for MLA, fla's merged KCP for KDA | `k3_pr_d_cp_ulysses` |

RFC #3029 carries the design discussion for all four; we are updating that
thread rather than opening a second RFC, since K3 shipping Block AttnRes is
exactly the adoption gate #3029 set.

None of the four touch the eager forward path. Three questions below are the
ones that decide how cleanly the two can meet.

---

## Comment 2 (on `model.py`, the decoder loop): factor the loop over a layer range

**Ask:** could the decoder forward be factored so the layer loop runs over an
arbitrary contiguous range and takes/returns its carried state, rather than
always iterating all layers from a single entry point?

**Why:** Block Attention Residuals thread a second value alongside the hidden
state -- a stack of committed block residuals -- which is read and appended to
as layers execute. Under pipeline parallelism a stage owns a slice of the
layers, so the loop has to be enterable at layer `i` with `(x,
block_residual_TND)` and exitable at layer `j` returning the same pair. With
the loop welded to "all layers, hidden state only", PP support means
duplicating the loop body rather than reusing it.

This is the single interface that decides whether our PP work is a patch on
yours or a fork of it.

## Comment 3 (on `state_dict_adapter.py`): a layout hook for grouped-GEMM experts

**Ask:** is there room for a hook in the expert state-dict path that lets a
backend declare per-expert <-> grouped layout, instead of the mapping being
fixed?

**Why:** the released checkpoint stores experts per-expert; a grouped-GEMM MoE
wants them stacked. Both directions have to work for load and for save, and
today the choice is baked in. A hook keeps one adapter for both.

## Comment 4 (on `parallelize.py`): the unsupported-parallelism guard

The guard reads as a list of `(name, enabled)` pairs raising on any hit. If
support lands piecemeal (say TP before CP), would you take the guard becoming
per-feature so a partially supported model does not have to edit one list? Minor,
but it is the thing we would touch first in every follow-up PR.

---

## Comment 5 (on `distributed/fsdp.py`): `add_zero_valued_dependency` -- we hit
## this exact class of bug, and the docstring understates the reach

Strong agree with this helper, and the docstring's reasoning is correct.

Two data points from running the same architecture under more parallelisms,
in case they are worth folding into the docstring or a test:

1. **It is not only "a batch with no images".** Under context parallelism the
   sequence is sharded, so a rank can hold a slice containing zero vision
   sentinels while every rank still receives the whole image batch. Same
   failure mode -- a module executed on a strict subset of the group -- reached
   by a different route, and one that only appears at cp>1.

2. **The failure is a hang, not an error.** We hit the analogous case in our own
   code (an `all_reduce` used to locate a CP rank's feature slice) and the first
   symptom was a step that never completed. Worth saying explicitly in the
   docstring, because "deadlock the step" reads as a possibility rather than the
   observed behaviour, and a hang sends people looking in the wrong place.

Happy to contribute a cp>1 regression test for this if useful.

---

## Comment 6 (on `config_registry.py` / `README.md`): naming, since the folder is shared

We renamed our own identifiers to `kimi_k3_*` this week, but deliberately kept
`kimi_linear_*` for the Kimi Linear paper's Table 2 scaling-law rows and for the
released Kimi-Linear-48B, on the grounds that renaming those would attribute a
real published model to K3 (there is no K3 at 48B -- K3 is 2.8T-A50B). If you
have a convention in mind for the shared folder we will follow it; flagging it
now is cheaper than resolving it in a rebase.

---

## Things we are NOT asking for

- We are not asking you to take our PP cross-stage adapter as a generic
  mechanism. That was proposed upstream before and declined; it stays private
  inside the model folder.
- We are not asking for design changes to the eager reference itself. The three
  asks above are all "leave a seam", not "change the math".

---

## Offer

If any of these are wanted, we can send them as separate follow-up PRs against
this one's layout: TP+KDA DTensor, EP + grouped-GEMM, PP + Block AttnRes, and CP
(Ulysses for MLA, fla's merged KCP for KDA). Each is independently testable and
none of them touch the eager path.
