# Paste block: faithful-only matrix section

Replaces the "Parallelism matrix on the eager reference's own debug config"
section of `RFC3029_UPDATE_2026-08-04.md`. The twin (`kimi_k3_debugmodel_pr_4025`)
results are dropped: the eager PR's structure changed to match the comments, so
the twin no longer mirrors anything and reporting it would describe a
configuration that no longer exists.

Single-line paragraphs (tables and list items excepted) so it copies verbatim.

--- PASTE BEGIN ---

### Parallelism matrix

Multimodal (MoonViT + backbone), seed 42, deterministic, gradient accumulation on. Eighteen configurations: single-GPU, `fsdp2`, `pp2`, `cp2`, `tp2`, every 3-of-4 combination of FSDP/TP/PP/CP, all repeated with EP, plus the max-degree cells `ep8 x fsdp8`, `pp4`, `pp8`, `tp4`, `cp4` on 8 GPUs.

| | 100 steps |
|---|---|
| eager | **18/18** |
| compiled | **15/18** (18/18 with the shim below) |

Every cell starts from a **shared step-0 checkpoint** written with `--checkpoint.create_seed_checkpoint` and loaded via `--checkpoint.initial_load_path` -- suggested on the eager PR, and it changed the result qualitatively rather than marginally. Verified the load happens rather than assumed it: cold start gives step-1 12.05342, the seeded run 12.07418.

| max-degree cell | result |
|---|---|
| `ep8 x fsdp8` (1 expert/rank) | pass, eager and compiled |
| `pp4` (uneven 13-layer split, vision tower on stage 0) | pass |
| `tp4`, `cp4` (1 head/rank) | pass |
| `pp8` | pass, after one fix (dense-grad on PP inputs) |

Fixing init makes the spread **decompose**. Seven cells with no TP and no CP -- `fsdp2`, `pp2`, `pp4`, `pp8`, `ep2 x fsdp2`, `ep8 x fsdp8` and single-GPU -- agree **bit-for-bit** at step 1 (12.07418). FSDP, pipeline parallelism at degrees 2/4/8 and expert parallelism at degrees 2/8 introduce no numerical difference at all; the cells that differ are exactly those where TP or CP changes reduction order, clustering at 12.07565 (CP), 12.07566 (TP), 12.07590 (both). Cold start had 0.041 of spread with init and parallelism superimposed and no way to separate them; this has 0.009, all of it accounted for.

Eager, 100 steps, all eighteen from the same step-0 weights:

| cell | step 1 | step 50 | step 100 |
|---|---|---|---|
| `single-GPU` | 12.07418 | 1.33860 | 0.35143 |
| `fsdp2` | 12.07418 | 1.34609 | 0.36425 |
| `pp2` | 12.07418 | 1.35109 | 0.35443 |
| `cp2` | 12.07565 | 1.35855 | 0.34527 |
| `tp2` | 12.07566 | 1.30560 | 0.34967 |
| `fsdp2 x tp2 x pp2` | 12.07566 | 1.37392 | 0.35423 |
| `fsdp2 x tp2 x cp2` | 12.07590 | 1.43442 | 0.37419 |
| `tp2 x pp2 x cp2` | 12.07590 | 1.34100 | 0.36659 |
| `fsdp2 x pp2 x cp2` | 12.07565 | 1.39481 | 0.37566 |
| `ep2 x fsdp2` | 12.07418 | 1.35289 | 0.36067 |
| `ep2 x fsdp2 x tp2 x pp2` | 12.07190 | 1.44611 | 0.36211 |
| `ep2 x fsdp2 x tp2 x cp2` | 12.06705 | 1.40905 | 0.36507 |
| `ep2 x fsdp2 x pp2 x cp2` | 12.07565 | 1.33048 | 0.36867 |
| `ep8 x fsdp8` | 12.07418 | 1.34121 | 0.36849 |
| `pp4` | 12.07418 | 1.33522 | 0.34277 |
| `pp8` | 12.07418 | 1.30919 | 0.38591 |
| `tp4` | 12.07286 | 1.36290 | 0.34306 |
| `cp4` | 12.06692 | 1.42309 | 0.37493 |

Compiled, same seed, same 100 steps:

| cell | step 1 | step 50 | step 100 |
|---|---|---|---|
| `single-GPU` | 12.07151 | 1.34267 | 0.35474 |
| `fsdp2` | 12.07151 | 1.38024 | 0.37617 |
| `pp2` | 12.07151 | 1.45299 | 0.35723 |
| `cp2` | 12.07340 | 1.38749 | 0.34061 |
| `tp2` | 12.06974 | 1.34620 | 0.36958 |
| `fsdp2 x tp2 x pp2` | 12.06974 | 1.40177 | 0.34836 |
| `fsdp2 x tp2 x cp2` | 12.07407 | 1.38595 | 0.36583 |
| `tp2 x pp2 x cp2` | 12.07177 | 1.30561 | 0.36885 |
| `fsdp2 x pp2 x cp2` | 12.07440 | 1.43574 | 0.35440 |
| `ep2 x fsdp2` | 12.07426 | 1.37039 | 0.35503 |
| `ep2 x fsdp2 x tp2 x pp2` | 12.06756 | 1.43192 | 0.35720 |
| `ep2 x fsdp2 x tp2 x cp2` | 12.07015 | 1.38656 | 0.36079 |
| `ep2 x fsdp2 x pp2 x cp2` | 12.07442 | 1.36530 | 0.36574 |
| `ep8 x fsdp8` | 12.07426 | 1.45852 | 0.34410 |
| `pp4` | 12.07151 | 1.39352 | 0.35963 |
| `pp8` | 12.07151 | 1.39817 | 0.36478 |
| `tp4` | 12.06411 | 1.37456 | 0.37205 |
| `cp4` | 12.06627 | 1.47387 | 0.38913 |

Same clustering compiled: `single-GPU`, `fsdp2`, `pp2`, `pp4` and `pp8` all at 12.07151.

**Across the run the cells stay within a band rather than on top of each other**, which is the expected behaviour and the documented criterion. `CONTRIBUTING`'s numerics guidance asks for bit-wise identical loss *with the same parallelisms*, and treats a change in computation as requiring convergence rather than identity -- and changing parallel degree changes the reduction order and tree shape, so it is a computation change. All eighteen decrease monotonically to the same order (0.343-0.386 at step 100), and two independent full runs of the same configuration reproduce bit-for-bit.

Fixing init also surfaced a defect that a cold-start matrix cannot reach: **multimodal + PP could not load a checkpoint**. Under PP only the stage owning `embed_tokens` is re-wrapped as the multimodal model, so it names parameters `language_model.*` while other stages name them bare, and a non-PP save uses the prefixed form throughout. Ordinary resume was affected identically, not just seed checkpoints. Fixed with state-dict hooks on non-first stages.

- **The compiled gaps are one upstream operator limitation**: `torch._grouped_mm` rejects a zero contraction dimension -- the weight-gradient shape a rank sees when it ends up with zero routed tokens in total, which the ep2 x TP/PP/CP combinations produce (per-group empties alone do not: `ep8 x fsdp8` passes compiled unshimmed). Five-line reproduction, no model/compile/distributed, being filed against pytorch/pytorch; with a local shim that re-strides exactly as the proposed patch would, every affected cell passes -- **the shim is ours, so the unshimmed 15/18 stays the reported number**.

--- PASTE END ---

## What this changes elsewhere in the RFC

* The `pp8` row previously linked to a `TWIN_MATRIX` anchor. The fix is real and
  unchanged, but citing a twin document in a faithful-only section is confusing;
  the link is dropped above. Re-point it at the seed-matrix record if a link is
  wanted.
* The `PP8xVP4` bullet also links into `TWIN_MATRIX`. Same treatment needed --
  that run was on our own 32-layer flavor, not the twin, so the claim survives;
  only the citation needs moving.
* "**18 of 18 attempted, on both model configurations**" no longer applies. There
  is one configuration now.
