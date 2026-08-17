# The 20-item review: what each one turned out to be

Branch `dep_exp_impl`. One row per item, with the evidence that decided it. Two of
the twenty were wrong as stated, and one of those two was wrong in a way that
uncovered something bigger than the item itself.

Read the two entries under "What the review could not see" first. They are not
review items; they are what checking the items exposed about the gate.

## What the review could not see

**The 58-cell gate never enters delta mode -- which is not the same as delta mode
being untested, and the difference is worth stating precisely because I got it
wrong first.** `adapter_enabled()` reads `attn_res_cache`, which defaults to False,
and no matrix script sets `TORCHTITAN_ATTNRES_CACHE`. So every PP cell in the gate
ships the full block stack on each hop (naive passthrough, numerically correct) and
"58/58" is not a regression statement about `layout.py`, the capture/augment grad
bridge, or the rank-local cache -- the code A2, B3 and B4 changed.

Delta mode itself has a lot of history: phase3 carries adapter/naive twins at 4 and
8 GPUs and at 48B pp8xvp4 and pp8xvp3, phase4 launches with it on by default, and
`PP_STATUS_2026-08-14.md` recorded **twelve delta cells all bitwise equal to
naive**, the densest being `ep2_fsdp2_tp2_pp2` on the LoRA flavor -- LoRA x EP x TP
x PP x DEP x dynamic CP with the cache on, equal to the same cell with it off. So
DEP and the cache coexisting (A3's configuration) is already covered, and the
adapter's numerics are well established.

What was missing is that those twelve cells were a hand-driven run rather than a
command, so nothing makes them repeatable after a change to that code.
`matrix_scripts/run_attnres_cache_cells.sh` is now that command: the same four PP
cells, taken from `run13_flav.sh` through `run_cells.sh` rather than retyped, each
run cache-off and cache-on with the gate's multimodal knobs exported. It also
carries one geometry that genuinely has never run:

* `multicommit` -- pp2 x vp2 with `layers_per_stage=8` over
  `attn_res_block_size=1`, so every stage commits eight blocks. This is exactly
  what A2's restriction refused, so no run of any kind has entered it.

Each cell is judged against its cache-off twin and must also show the wrap line. That
assertion earned its place immediately: the FIRST version of the script reused the matrix
PP cells through `run_cells.sh`, those cells run the default schedule, delta mode is gated
on `isinstance(pp_schedule, Interleaved1F1B)`, and so both halves of every A/B ran naive
passthrough. The losses matched perfectly while measuring nothing, and the wrap-line check
is the only thing that said so.

Two more attempts failed before one ran, and the reasons are worth keeping because each
looks like the thing under test:

* `BlockLayoutTables` needs `n_layers % num_stages == 0`, and the multimodal pp8vp4
  flavor's 30 layers are not divisible by 4;
* that flavor is not an AttnRes model at all -- `num_blocks` is None, so the adapter passes
  through no matter the schedule;
* `first/last_stage_less_layers` default nonzero, which makes the split uneven, and the
  adapter's contiguous-layout check then refuses with "layer 24 sits on stage 2";
* the 48B L32_N8 carrier satisfies the arithmetic but OOMs with 16 layers on each of two
  ranks.

Hence `kimi_k3_mini_attnres_multicommit` (16 layers in 8 blocks of 2), added for the same
reason `report_arch_pp8vp4` exists: so the geometry is expressible.

**And the judge itself was wrong at first.** It required bit equality, and delta mode is
NOT bit-identical to naive -- the mid-stage rebuild of the block stack is "the only
reorder point" (`attn_res_model.py`), so the summation order differs and the curves
separate from step 2 onward. Step 1 agreeing to the digit is what says the forward is
unchanged. Measured:

| geometry | wrap | max abs dLoss over 6 steps |
| --- | --- | --- |
| single, 1 commit/stage, pp2 x vp4 | 4 stages | 0.01981 |
| multi, 2 commits/stage, pp2 x vp2 | 2 stages | 0.00828 |

Both within phase3's recorded `|dLoss| <= 0.011` neighbourhood, and multi-commit is
*better* than single-commit, which is the claim removing the restriction rests on. Neither
run logged a capture-count mismatch (B3, now a hard failure) or a step-end slot sweep
(B4). Had the bit-equality judge stood, it would have condemned a correct change.

**Every CP cell was running the wrong CP.** See B1.

## A

**A1 -- real, fixed (`8a6a86f2a`), and its stated impact was wrong.**
`apply_ep_kimi_k3` read `layer.ffn` after the attribute became `layer.moe`, so it
logged "EP plan wrapped 0 MoE layer experts" through every green EP cell. Fixing
the name produced "MoE has already been parallelized", which is how the real
caller surfaced: the declarative driver reaches the MoE through the layer's own
`Module.parallelize`. EP was applied all along; the reporting was broken. Now
`verify_ep_applied` asserts each layer's routed-expert parameters are DTensors
with a non-replicate placement.

**A2 -- real, fixed, and the restriction was stale rather than unimplemented.**
The layout refused a stage committing more than one block, pointing at per-block
consumer lists in `_RecvBlockGradsFromConsumers`. That class was deleted in
`89868bde5` when the custom grad P2P was replaced by the rank-local capture and
augment hooks. The table it protected (`consumer_stages_of`) had no readers left
anywhere in the repo, and neither did `_grad_tag_base`. Every remaining table is
keyed by the commit's index within its producer stage, and so is the runtime: the
cache stores `(rank, stage, block_idx_in_producer)`, `_finish_forward` installs one
hook per commit, and `expected_same_rank_captures` takes the local index. Removed,
with eight unit tests over multi-commit layouts. The review's arithmetic was right
-- pp=2, V=2 at 93 layers and `attn_res_block_size=12` does hit it -- but the fix
it proposed (a nicer error earlier) would have entrenched a limit that no longer
existed.

**A3 -- the ordering risk as stated does not hold; the joint invariant is worth a
cell, and now has one.** The order is currently correct and not by accident:
`_install_vision_prefetch` runs at line 1595 of `pipeline_adapter.py` and
`_install_step_drop_patch` at 1676, so the adapter's sweep wraps outside DEP's
drain, and `step` unwinds drain-then-sweep. But the review's failure mode -- "drain
depends on the tower's forward graph, which the cache clearing may already have
affected" -- is not reachable either way: the tower's graph is held by the
`GradQueue`'s own `(output, grad)` references and the cut has already severed it
from the text graph, while `_drop_all_cached_and_clear` only drops block tensors
and slots. The two `finally` blocks act on disjoint objects. So no assertion was
added to protect a risk that is not there (see the repo's rule against speculative
defensive checks). And the two mechanisms coexisting is not unevidenced: four of the
twelve delta cells in `PP_STATUS_2026-08-14.md` are delta x DEP x dynamic CP, with
four more adding LoRA, all bitwise equal to the cache-off twin. What that run did
not have is a repeatable form, which is what `run_attnres_cache_cells.sh` gives it.

## B

**B1 -- the mutual exclusion is impossible, and asking about it uncovered that
every CP cell tested the wrong path.** Ulysses and KCP cannot both apply to one
KDA layer: `kda_cp_mode` is a single enum field and `forward` dispatches on it. And
the "both at once" the review feared is in fact the design -- a CP run is KCP on
the KDA layers AND Ulysses on the MLA layers, together, because KCP decomposes the
delta-rule recurrence and says nothing about softmax attention.

What the wiring really got wrong is that it did not know which mode was in play. It
enforced the Ulysses head-divisibility rule on KDA layers even under KCP, where
heads are never split (so it rejected configurations that work), and it logged
"Applied Ulysses CP" on a KCP run -- the same stale-report shape as A1.

Behind that: `kda_cp_mode` defaulted to `"ulysses"`, and no matrix cell overrode
it. The gate has seven CP cells across three arms, so **21 of 58 cells reported CP
passing while running the path the report does not describe** (sec 5.1.2 is KCP)
and the one that cannot reach the target context length, since Ulysses gives every
rank the whole sequence for its head subset and activation memory does not fall
with cp.

Fixed in three parts. The default is now `"kcp"`; Ulysses for KDA keeps an explicit
A/B flavor. The wiring checks per mode (head divisibility for Ulysses, fla's CP ops
importable for KCP) and names the mode it wired:

    Applied CP cp_degree=2: 4 MLA layer(s) Ulysses, 9 KDA layer(s)
    kda_cp_mode=kcp (13 attn layers total)

And the batch axis had to be dealt with before the default could move: fla's
`causal_conv1d_cp` asserts `[1, T, D]`, so the CP path raised for B > 1, which is
most of the gate's cells. `_forward_kcp` now loops the batch. Flattening it into
one packed sequence would be wrong rather than merely awkward --
`build_cp_context` cuts the GLOBAL packed sequence into contiguous rank-ordered
pieces, while a rank holds piece r of EVERY sequence, so the layouts coincide only
at B = 1 -- and the loop is also what the recurrence needs, since delta-rule state
must not carry from one sequence into the next. The cost is B fixed-size
prefix-scan all-gathers instead of one, with B equal on every rank.

One check was deliberately NOT added: a `local_batch_size == 1` guard at wiring
time. What KCP cannot take is a batch axis on the tensor the module sees, and that
is the micro-batch, not `training.local_batch_size` -- under PP the two differ by
the micro-batch count. `_forward_kcp` checks the real B. A less accurate copy
earlier would reject configurations that run, which is the same class of mistake as
the head-divisibility rule it replaced.

Evidence, from `matrix_scripts/kcp_batch_parity.py` rather than from a loss curve,
because dp1-vs-cp2 losses are both plausible: each rank's CP output against the
single-rank forward's corresponding slice, per batch row, at B = 2. Relative error
`4.310e-03` on rank 1 (`row0=4.310e-03 row1=4.132e-03`) and `4.660e-03` on rank 0,
bf16 noise, rows independent. Rank 1 is the load-bearing one -- rank 0's incoming
recurrent state is zero and its conv needs no halo either way, so a probe reporting
only rank 0 would pass with the prefix scan and the halo both broken.

**B2 -- real, but not KCP's and not a hole KCP opened.** `build_kcp_context` did
hardcode `cu_seqlens=[0, total]`, i.e. one document. It now takes `cu_seqlens`,
defaulting to that. The important part is the scope: nothing in this repo hands KDA
document boundaries in ANY mode -- both non-CP call sites pass `cu_seqlens=None` to
`chunk_kda` -- so a packed SFT batch already carries delta-rule state across
document boundaries with or without CP. Refusing packed-plus-KCP specifically would
single out one combination while the identical defect stayed silent in the others.
Fixing it properly means threading the dataloader's boundaries through every KDA
call site, which is its own piece of work and is now written down as such.

**B3 -- real, fixed.** A capture-count mismatch warned and let the step apply an
incomplete gradient for that block. It raises now. The count comes from the static
layout, and delta mode is gated on `isinstance(pp_schedule, Interleaved1F1B)`,
under which every same-rank consumer's backward precedes the producer's own -- so
there is no configuration where continuing is right, and no flag was added to allow
it. Under LoRA the block tensor can arrive without `requires_grad`, in which case no
hook is installed and nothing is checked, which is correct rather than a gap.

**B4 -- real, fixed.** The step-end sweep now clears the captured-grad slot tables
outright and reports how many it cleared. The mb-keyed drop only reached slots whose
micro-batch still had cached blocks, so a backward that died mid-micro-batch (OOM
being the ordinary cause) leaked a grad tensor per slot. Outside the exception path
the count is zero, so a nonzero one in a log is a real signal that an mb-end
assertion did not run.

**B5 -- real, fixed (`964e18bcd`).** `GradQueue` now takes `max_pending` and runs the
earliest entry in place once the bound is reached, so the memory window is a configured
quantity rather than whatever the plan implies. Zero (unbounded) stays the default on
purpose: the window is unmeasured (`DEP_BUBBLE_STATUS_2026-08-16.md`, open item 3), and
a nonzero default would swap a known behaviour for a guessed number. Four unit tests,
including that the bound changes when rather than whether a gradient runs.

**B6 -- real, fixed (`2f768b4a3`).** `verify_params_distributed` runs after the leftover
sweep, which is the last thing that distributes anything, and refuses to continue if any
parameter is still a plain Tensor -- naming the parameters instead of letting
`clip_grad_norm_` report `aten._foreach_mul_.Tensor got mixed` with no idea which one or
which mechanism. Only DTensor-ness is asserted, not the mesh: parameters legitimately
live on more than one mesh, so a whitelist would encode the layout the sweep exists to
arrange. The mesh histogram is logged instead.

## C and D

**C1 -- all three points are already implemented; this one is a false positive.**
Checked rather than taken on trust, since it was described as the item that decides
whether an MFU number is credible.

* KDA is already counted as linear: a separate `12 * n_kda * kda_num_heads *
  kda_head_dim**2` term with no `seq_len` in it.
* AttnRes already has its own term, `6 * (2 * n_layers + 1) * (num_blocks + 1) *
  hidden_size`, for the source mixing.
* LatentMoE needs no special term. Expert weights are shaped by the latent
  dimension, so the `6 * activated_params` term carries it; the latent projections
  themselves land in `dense`, which is correct because every token passes through
  them.

The MLA term uses `6 * n_full_attn * n_heads * head_dims * seq_len`, which is
exactly upstream's own `get_moe_model_nparams_and_flops` (`torchtitan/models/
utils.py`), so it is not the factor-of-two error a `6` there can look like.

Bucketing verified on `kimi_k3_debugmodel_report_arch` at seq 4096: routed
16.52M / router 0.02M / latent 1.58M / dense 60.46M / embedding 41.94M, i.e. the
routed experts are found by name rather than falling into `dense` and being
counted as fully activated -- the failure the code comments record having happened
once. Terms: linear 0.397G, MLA 0.031G, KDA 0.007G, AttnRes 0.0002G per token.

**C2 -- real, fixed (`964e18bcd`).** `run_next` counts the slots that find nothing, and
the per-step report now carries four figures: ran-at-a-planned-slot,
drained-at-step-end, forced-by-the-memory-bound, found-nothing. A high idle count is the
signal to turn the greedy `min(pending)` into any-pending.

**C3, C4 -- open. The reasoning in the review is right, including why the ratio
must be a parameter** (a plan derived from each rank's own timing stops being
identical across ranks). Worth adding: this session already paid for the failure
mode, by passing a ratio measured at seq 4096 to a seq-256 cell where the true
value is about 28x larger.

**C5, C6 -- open, both profiling questions, correctly ranked low.**

**D1, D2, D4, D5 -- done (`c95e2703e`), and the folder README was worse than the review
said.** It carried "CP is out of scope: KDA needs Ulysses-style head sharding or
LASP-style cross-rank state passing, neither of which fla-core provides today" while
seven CP cells across three arms were running. A reader was being told the feature does
not exist. Rewritten to what runs, including that the two modes apply to disjoint layers
and run together. D1 and D2 became one entry, since both are the `tp*cp` quotient seen
from two sides. D4 states the unit of `dynamic_cp_min_patches` (pre-merge patches, so
256 is a 16x16 grid = a 224x224 image = 64 tokens post-merge) and says plainly that it
is not calibrated. D5 documents V=1 as supported: with no second virtual stage there is
no cached prefix to diff against, so the naive chain relay is the bandwidth lower bound
and delta mode has nothing to omit.

**D3 -- done (`c95e2703e`).** Declared in the README with the pointer to MoonEP, and with
the review's sharper second point kept: at 896 experts with top-16 the report itself
(sec 2.3) puts the sparsity beyond where fixed-step auxiliary-loss-free bias updates are
known to behave, so the gap is not only throughput. `quantile_balance.py` addresses the
router half (it solves for the bias instead of nudging it) and does not make the
transport balanced.

## Two defects the review did not contain

Both came out of doing the review rather than out of reading it, and both were found
by the gate -- which came back 45 of 58 on the first run after the KCP switch.

**KCP had never run under TP.** Every failing cell that had both TP and CP died inside
fla's triton kernel with `CUBLAS_STATUS_INTERNAL_ERROR` or an illegal memory access,
while every CP cell WITHOUT TP passed. KDA is `NoParallel` under TP, so its short-conv
weight is a `DTensor(Replicate)`, and `conv_with_halo` handed it straight to
`causal_conv1d_cp`. The Ulysses path unwraps the same weight in its own `conv_subset`;
the KCP path never did. It could not have been noticed before, because the only flavor
that selected KCP ran without TP -- so changing the default is what exposed it, and the
bug was strictly older than the change that surfaced it. Fixed in `2f768b4a3`, verified
on all five failing cells across both the text and the multimodal flavor.

**The EP verifier read a rank-local value under a job-level guard.** `ep_expected` is
assigned only when this rank has MoE layers; the `verify_ep_applied` call was guarded on
`parallel_dims.ep_enabled`, which is a property of the job. Under PP a rank can hold only
the vision-tower stage, so on the multimodal arms every `ep+pp` cell raised
`UnboundLocalError`. This one I wrote, in the A1 fix, one commit earlier -- and the gate
run that blessed A1 predates it, so nothing had exercised it. Fixed in the same commit.

Neither is a review miss: the review read code, and neither of these is visible by
reading. The first needed a configuration nobody had run, the second needed a rank that
holds no MoE. That is the argument for running the whole matrix per change rather than
the cells that look related.

## Status of the code

Four commits on `dep_exp_impl`, ending at `c95e2703e`: the KCP default with A2/B1/B2/
B3/B4, then the two gate-found regressions with B6, then B5 with C2, then the D-level
documentation.

**The gate is 58 of 58 at `c95e2703e`**, counted by the gate itself, after coming back
45 of 58 on the first run. Beyond the cell count, four things it confirmed:

* the three cells that died with illegal memory access and the four that died with
  UnboundLocalError all pass, on both the text and the multimodal flavors;
* no cell logged a capture-count mismatch, a step-end captured-grad slot, or a plain
  parameter at clip time -- the three canaries this round added;
* B6's mesh histogram came back `{"('tp',)": 250, "('ep',)": 30}` on the EP cells, which
  is the evidence for why it asserts DTensor-ness and not a mesh whitelist: parameters
  legitimately live on two meshes at once;
* the KDA mode line reads `kda_cp_mode=kcp` on every rank that holds KDA layers and `-`
  on the ranks that hold none, rather than inventing a mode for them.

What the gate still cannot prove is the first section of this file: it does not enter
delta mode, so A2, B3 and B4 are covered by `run_attnres_cache_cells.sh` instead, which
runs separately -- one GPU job at a time.

## The one item that needed a long run

The delta path differs from naive by max `|dLoss|` 0.0198 over six steps, and six steps
cannot tell noise from a slow divergence. That distinction decides whether the adapter's
-11.4% peak memory is a good trade, and nothing on record ran long enough: every
delta-vs-naive comparison in this logbook is tens of steps.

3000 steps per arm, `matrix_scripts/run_delta_convergence.sh`, pp2 x vp2 with two commits
per stage:

| arm | final loss | mean rel gap vs naive, first 10% -> last 10% |
| --- | --- | --- |
| naive (seed 42) | 1.77437 | -- |
| delta (seed 42) | 1.78017 | 0.00245 -> 0.00342 |
| naive, seed 7 | 1.76570 | 0.00465 -> 0.00555 |

**The transport's difference is 62% of the run-to-run spread of one configuration**, and
neither gap grows across the run. So the 0.0198 was noise being averaged away.

The third arm is the whole argument. Without it, "0.34% apart at the end" reads as
evidence; with it, that number is visibly smaller than what the same configuration does to
itself when reseeded, and therefore not attributable to delta mode at all. Any future
claim of the form "X changes the loss by Y%" on this stack needs the same floor measured
alongside it.

`attn_res_cache` stays False, and the reason is no longer trust. Engaging delta mode needs
Interleaved1F1B, `n_layers % num_stages == 0`, and an even split
(`first/last_stage_less_layers 0`). Most configurations satisfy none of those and would
keep taking the passthrough they already take; the ones that qualify would change
transport with no gate cell able to observe it, since the 58-cell gate never enters delta
mode. Flipping it is a launch-configuration decision with the numbers now attached, not a
correctness question.
