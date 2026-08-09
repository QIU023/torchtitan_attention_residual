# Overnight review pass: what closed, what did not, what to decide

Companion to `OVERNIGHT_REVIEW_2026-08-09.md` (the per-finding ledger). This is the
summary a reader needs first.

## Traceback

* Fork `attention_residual_dev`: started at **`3e7e23e3c0eb1085dd19fc53b3a5484ef8320eb4`**,
  now **12 commits ahead**, none pushed.
* Logbook: started at `d80b408100e6266b2d88ff641e9cec3b85c38080`.
* Disk: 299 GB free at close (82 GB used of 380 GB). Matrix artifacts are ~20 MB per run;
  per-cell checkpoints are deleted by the runner as each cell finishes.

The twelve fork commits, oldest first:

    fabbfc68d  nine findings, including three self-inflicted
    f3834c8b3  close the from_hf round trip
    ef5ddfc84  two defects an internal round trip cannot see
    acce7b0f3  carry the AttnRes block size instead of re-deriving it
    1b032ae1f  weight decay off 1-D params; QAT wrapper looks like a Linear
    81bcfb53d  a carve-out that carved nothing, a dead twin, a silent sweep
    a88cd727c  the unreachable layer->stage discovery becomes a guard
    48e691796  evict what the cache holds; a config error stays an error
    9bed2644b  count activated MoE params, not total, in flops_per_token
    c7048afe0  the expert GEMM seam, one vision splice, two non-bugs
    d5e3063bc  restore the fla-core import guard; drop an embed-path host sync
    549ef4a47  key the K3-shape predicate on the row; re-pin uneven-split video

## Read this before pushing anything

Those twelve messages cite findings as bare `#N`. On a branch belonging to a PR against
pytorch/torchtitan, GitHub resolves each one against THAT repo and writes a permanent
reference event into an unrelated third-party issue -- the mechanism that already fired ~14
times on PR-4025. Nothing is pushed, so nothing has fired. Rewording them needs a history
rewrite, which policy denied and I did not work around. Two ways out, in
`OVERNIGHT_REVIEW_2026-08-09.md` under the blocker heading; the later commits already use
`finding N`.

## Verification standard used

Every functional change ran the full multimodal 18-cell matrix
(`kimi_k3_debugmodel_report_arch`, 10 steps, seed 42, deterministic). **All seven matrix
runs came back 18/18 bit-identical** to `MATRIX_18_CORRECTED_2026-08-09`, with the three
TP+CP cells failing as the documented `KIMI_VIT_DYNAMIC_CP=0` limitation, unchanged.

Bit-identity is the right criterion for most of this batch and the wrong one for two
changes, which is worth separating:

* Changes that must not move the numbers (checkpoint mapping, dead code, accounting,
  guards): bit-identity is the result.
* `finding 21` (weight decay) genuinely changes training numerics, and the matrix cannot
  see it at all -- the matrix flavor configures a single AdamW group with pattern `.*` and
  never reaches `default_muon()`. Its judge is a 10-step A/B on `kimi_k3_mini_muon`:
  identical through step 2, diverging 1e-4 to 4e-3 from step 3, both arms converging alike.
  The matrix run for it is a control (nothing leaked into the default path), not a judge.
* `finding 22` (QAT hiding `.weight`) is likewise outside the matrix. Its judge is tag
  counts: 8 projections tagged without QAT, 0 with it, 8 again after the fix.

Suite went 315 -> 345 passing, 1 skipped. Every fix that could have a negative control got
one, and two of those controls had to be redone because the first version failed for the
wrong reason (see below).

## The one that changes published numbers

`findings 45/53/73`: all four MoE parameter buckets in `get_nparams_and_flops` matched no
parameter name at all -- the real names are `ffn._moe.routed_experts.inner_experts.*`,
`ffn._moe.router.gate.weight`, `ffn._moe.shared_experts.*`, and the code looked for
`.moe.experts`, `.moe.router`, `.moe.shared_experts`. Every MoE parameter fell into `dense`
and was counted as activated, so **flops_per_token reported total-as-activated**.

Corrected, the K3-shaped `2p8t` flavor reports 104.19B activated linear parameters against
K3's published **104.2B activated**; the broken version reported 2.78T, the total. That
external agreement is what makes the direction trustworthy.

| flavor | broken | fixed | overstated |
|---|---|---|---|
| `report_arch` (matrix) | 0.4806 G | 0.4063 G | 1.18x |
| `bubble_ratio` (the DEP latency runs) | 0.8044 G | 0.6248 G | 1.29x |
| `debugmodel_block_attn_res` | 0.083 G | 0.041 G | 2.02x |
| `2p8t` | 16689 G | 645 G | 25.89x |

`VIT_DEP_DESIGN_2026-08-07` and `HANDOFF_2026-08-08_EVENING` now carry the correction
inline. The widely-cited `mfu 0.12%` reads **~0.093%**. My first annotation said "about 2x"
everywhere, which was wrong -- that is one flavor's factor, and the cited runs are a
different flavor. No conclusion in either document depended on MFU being as high as 0.12%,
and a lower MFU strengthens "the encode is already free, a run-ahead cannot help".

## Three findings whose premises did not survive checking

Recorded because "fixed" would have been the wrong answer, and in one case I had already
written the fix:

* **`finding 64`** (pack_video ignores frames after the first): unreachable.
  `prepare_image` derives its grid from `navit_resize(W, H, ...)`, a pure function of the
  dimensions, and `pack_video` takes one stacked `[F, C, H, W]` tensor -- so all frames
  share H and W. The finding assumed ragged input, which is what `pack_images` takes. I had
  the validation written before I checked; it was dead code, so I reverted it and documented
  why the constraint holds instead.
* **`finding 35`**'s substantive half (grafted `init_weights` leaves the tower at
  `torch.empty`): absent. The class defines its own `init_weights`, which does init the
  tower, and the graft loop's `hasattr` guard always skipped that entry. Measured, not
  argued: two same-seed builds agree on all 30 tower/projector parameters, all finite. Its
  code-quality half is real and fixed.
* **`finding 28`** (`is_mla` constant-True, delete the dispatch): a config with all five MLA
  dims None and `mla_use_nope` False constructs fine, so the branch is live and deleting it
  would drop a guard. Only the exception type was wrong; it is a `ValueError` now.

And one that is real as code but unreachable as a failure: **`finding 63`**. Fixed anyway,
because making the comparison well-scoped costs nothing, but recorded as not-biting.

## Open, with reasons

| what | why not closed |
|---|---|
| `finding 44` (MTP + chunked loss) | needs an `mtp_loss` redesign, not a local fix |
| `finding 32` (env-var topology) | upstream will not take env vars; migrating to config fields is its own change |
| `finding 60` | the fix is in but the numbers never moved on any configuration tried, so it rests on the code reading alone |
| `finding 66` (general case) | the only divergence source I could name is closed; a setup-time all-reduce would defend a case I cannot construct |
| TP x dynamic CP defect 2 | needs the `wo` local/DTensor contract settled first |
| DEP 30-layer 6e-4 divergence | the unchunked control is blocked by an FSDP uniform-gradient-dtype assertion |
| `kimi_k3_mini_block_attn_res` name collision | a trainer config function and a registry flavor share it; fixing it is a rename touching launch scripts, i.e. a decision |
| TIER 7 reuse (54/56/55/58/57/6/7) | 180-line `apply_fsdp` fork, SDPA and AdamW duplication, the MLA and FeedForward difference tables. Large, and `55`/`58` would move numerics |
| TIER 9 remainder (13/14/15/16) | efficiency: stack rebuild on the hot path, per-head Newton-Schulz, `cu_seqlens.tolist()` sync, one all_reduce per MoE layer |
| TIER 10 (12/11/4/5) | comment triage across 48 blocks, and the pre-rule commit trailers |
| Core-footprint audit (1/2/3) | delete `grouped_mm_empty_shim` from the fork; decide whether the `pipeline_parallel.py` microbatch fix is an upstream bug; `models/utils.py` placement skip is a small core PR candidate |
| HF-reference parity test | still the missing judge for `finding 62` (bicubic antialias), which has only a no-regression guard |
| Core grad-norm dtype fix | in `torchtitan/distributed/utils.py`, a genuine core defect. My recommendation is a separate small upstream PR; awaiting the call. **Local claude needs to know this file is modified.** |

## Harness

`run13_flav.sh` assigned fixed per-cell ports and never retried, so a socket left in
TIME_WAIT by the previous matrix run killed a cell. Three consecutive runs each lost exactly
one (`ep2_fsdp2`, `tp2`, `fsdp2_pp2_cp2`), and each reported `FAIL (0/10)` -- indistinguishable
from a numerical regression until you open the log. It retries on a fresh port now, which
`run_maxdeg.sh` already did; the last run lost nothing.

Two harness lessons worth keeping:

* When rerunning one cell by hand, the GPU count comes from the parallelism product, not the
  cell name. `fsdp2_pp2_cp2` on 4 GPUs gives
  `Invalid parallel dims: dp_replicate(1) * dp_shard(2) * cp(2)`.
* `rerun_failed.sh` hardcodes `FLAVOR=kimi_k3_debugmodel_pr_4025`. Using it to rerun a cell
  from a `report_arch` matrix silently compares two different flavors.

## Two negative controls that had to be redone

Both times the first version failed for the wrong reason, which reads as success:

* `finding 65`: reverting the whole change made both new tests fail -- with `AttributeError`
  from the method rename, not on behavior. Reverting only the sweep body and keeping the new
  name isolates it: the forward-only test fails, the backward-marked one still passes.
* `finding 34/69`: reverting to HEAD would have failed on the extracted helper not existing.
  Reverting only the three-line loop to the discarding version gives the real control -- all
  three ops SUBFAIL.

The general shape: a control that fails because the code is missing tells you nothing about
whether the code is right.
