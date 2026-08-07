# Matrices from a shared step-0 checkpoint

The eager PR's author suggested fixing the init weights. Doing so changed the
result qualitatively, not just quantitatively, and it exposed a defect that a
cold-start matrix structurally cannot see.

Protocol: one seed checkpoint per flavor, written on a single GPU with
`--checkpoint.create_seed_checkpoint`, then every cell loads it via
`--checkpoint.initial_load_path`. The two flavors need separate seeds -- their
layer composition differs by one layer.

Verified the load actually happens rather than assuming it: cold start gives
step-1 12.07003 and the seeded run gives 12.09025. Identical numbers would have
meant the path was silently ignoring the checkpoint, which is the failure that
looks like success.
(Those are the unchanged-architecture flavor's numbers; the report-architecture
flavor's pair is 12.05342 cold against 12.07418 seeded.)

## The defect it exposed: PP could not load any checkpoint

Six of eighteen cells failed immediately:

    RuntimeError: Missing key in checkpoint state_dict: final_attn_res_norm.weight

The key exists, as `language_model.final_attn_res_norm.weight`. Under PP only
the stage owning `embed_tokens` is re-wrapped as the multimodal model, so it
names parameters with that prefix while every other stage is the bare text
model and names them without it. A non-PP save uses the prefixed form
throughout.

So **multimodal + PP could not load any checkpoint at all** -- not a seed
checkpoint, and not an ordinary resume, which any real run needs. Every matrix
before this one cold-started, and a cold start loads nothing, so 18/18 was
achievable with this broken the whole time.

Fixed with state-dict hooks that add and strip the prefix on non-first stages
(commit `63774c66f`), rather than re-wrapping every stage: the wrapper's forward
expects a tower and an image-splice path, and giving middle stages one to settle
a naming question would trade a naming bug for a forward bug.

## What fixing init actually bought

Not a smaller spread -- a decomposable one.

`kimi_k3_debugmodel_report_arch`, eager, 100 steps, all eighteen from the same
step-0 weights:

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

**The step-1 values cluster by reduction structure, and the clusters are exact.**

| step-1 | cells |
|---|---|
| 12.07418 | `single-GPU`, `fsdp2`, `pp2`, `ep2 x fsdp2`, `ep8 x fsdp8`, `pp4`, `pp8` |
| 12.07565 | `cp2`, `fsdp2 x pp2 x cp2`, `ep2 x fsdp2 x pp2 x cp2` |
| 12.07566 | `tp2`, `fsdp2 x tp2 x pp2` |
| 12.07590 | `fsdp2 x tp2 x cp2`, `tp2 x pp2 x cp2` |

Seven cells with no TP and no CP agree **bit-for-bit** with the single-GPU run:
FSDP, PP at degrees 2, 4 and 8, and EP at degrees 2 and 8 introduce no
numerical difference at all. The cells that do differ are exactly the ones
where TP or CP changes the floating-point reduction order, and they group by
which of the two is present.

That decomposition is the thing cold-start could not deliver. Its spread was
0.041 with init differences and parallelism differences superimposed and no way
to separate them; here the spread is 0.009 and every part of it is accounted
for.

## Compiled, same seed, same 100 steps: 18/18

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

Same clustering: `single-GPU`, `fsdp2`, `pp2`, `pp4` and `pp8` all at 12.07151.

**This run has the grouped_mm shim installed**
(`matrix_scripts/gmm_shim.py`), which is ours and not upstream. Without it the
three EP cells that combine with TP or PP/CP fail on
`torch._grouped_mm` rejecting a zero-length contraction dimension, and the
matrix is 15/18. **15/18 stays the reported number for compiled**; 18/18 is
evidence about the operator.

## The unchanged architecture, eager, 10 steps: 18/18

`kimi_k3_debugmodel_pr_4025` from its own seed checkpoint. Same clustering
(`single-GPU`, `fsdp2`, `ep2 x fsdp2`, `pp8` at 12.09025; `pp2`, `ep8`, `pp4`
at 12.09026; the CP cells at 12.06796; the TP cells at 12.08052).

| cell | step 1 | step 5 | step 10 |
|---|---|---|---|
| `single-GPU` | 12.09025 | 10.83241 | 9.76928 |
| `fsdp2` | 12.09025 | 10.85353 | 9.76449 |
| `pp2` | 12.09026 | 10.82154 | 9.76697 |
| `cp2` | 12.06796 | 10.86433 | 9.77591 |
| `tp2` | 12.08052 | 10.84434 | 9.75798 |
| `fsdp2 x tp2 x pp2` | 12.08052 | 10.81674 | 9.75707 |
| `fsdp2 x tp2 x cp2` | 12.08781 | 10.78145 | 9.75632 |
| `tp2 x pp2 x cp2` | 12.08781 | 10.78699 | 9.75852 |
| `fsdp2 x pp2 x cp2` | 12.06796 | 10.86920 | 9.77296 |
| `ep2 x fsdp2` | 12.09025 | 10.85596 | 9.77358 |
| `ep2 x fsdp2 x tp2 x pp2` | 12.08608 | 10.78652 | 9.75803 |
| `ep2 x fsdp2 x tp2 x cp2` | 12.08889 | 10.82732 | 9.76955 |
| `ep2 x fsdp2 x pp2 x cp2` | 12.06796 | 10.87242 | 9.77903 |
| `ep8 x fsdp8` | 12.09026 | 10.84004 | 9.76476 |
| `pp4` | 12.09026 | 10.83241 | 9.76928 |
| `pp8` | 12.09025 | 10.82815 | 9.76795 |
| `tp4` | 12.07177 | 10.85678 | 9.78266 |
| `cp4` | 12.08145 | 10.78888 | 9.75687 |

## What is still not claimed

Cross-stack alignment. Fixing init makes it *possible* -- the obstacle was
never only the rebase, it was that two cold starts consume RNG differently --
but it needs the same weights loaded into both stacks, which needs our
state-dict adapter to round-trip against theirs. Not done, and not claimed.

The 100-step runs still overfit a smoke dataset. That has not changed.

---

## Dense control: the same matrix with MoE removed

`kimi_k3_debugmodel_report_arch_dense` -- one field changed from the report
architecture, `first_k_dense_replace` set to the layer count, so every layer is
a plain FFN. Same 13 layers, same KDA/MLA composition with the trailing Gated
MLA, same Block AttnRes, same vision tower, same data, its own seed checkpoint.
Verified zero MoE parameters remain; 106.5M total.

Thirteen cells, not eighteen: **expert parallelism is inapplicable** to a dense
model, so those five are reported as such rather than as failures.

| cell | step 1 | step 50 | step 100 |
|---|---|---|---|
| `single-GPU` | 12.06172 | 0.53980 | 0.11974 |
| `fsdp2` | 12.06172 | 0.54802 | 0.11066 |
| `pp2` | 12.06172 | 0.54380 | 0.13083 |
| `cp2` | 12.06102 | 0.59149 | 0.11894 |
| `tp2` | 12.06201 | 0.54532 | 0.12916 |
| `fsdp2 x tp2 x pp2` | 12.06201 | 0.51759 | 0.11495 |
| `fsdp2 x tp2 x cp2` | 12.06178 | 0.55177 | 0.11613 |
| `tp2 x pp2 x cp2` | 12.06178 | 0.54580 | 0.11683 |
| `fsdp2 x pp2 x cp2` | 12.06102 | 0.55344 | 0.11935 |
| `pp4` | 12.06172 | 0.55650 | 0.12217 |
| `pp8` | 12.06172 | 0.52187 | 0.10420 |
| `tp4` | 12.06203 | 0.54762 | 0.12445 |
| `cp4` | 12.06217 | 0.51810 | 0.13365 |

13/13, monotone throughout. The same clustering by reduction structure: five
cells agree bit-for-bit at step 1 (`single-GPU`, `fsdp2`, `pp2`, `pp4`, `pp8` at
12.06172), and the ones that differ are the TP and CP cells.

### What it establishes

The run-horizon band is not architecture-specific. Comparing the thirteen
non-EP cells on both models:

| horizon | dense abs spread | MoE abs spread | dense rel | MoE rel |
|---|---|---|---|---|
| step 1 | 0.00115 | 0.00885 | 0.01% | 0.07% |
| step 50 | 0.07390 | 0.12882 | 13.6% | 9.5% |
| step 100 | 0.02945 | 0.04314 | 24.5% | 12.0% |

Absolute spread is the same order on both and slightly smaller on dense.
Relative spread reads higher on dense only because it converges three times
further (0.120 against 0.360 at step 100), so an equal absolute difference is a
larger fraction of a smaller loss.

So the band tracks reduction order, which changes with parallel degree, rather
than anything about routing. That is the documented expectation: bit-wise
identity is required *with the same parallelisms*, and a change of parallel
degree is a change of computation, judged on convergence.

This control was run specifically to test an attribution that appeared in an
earlier draft of the RFC -- that the band came from MoE top-k flipping. It does
not; the dense model shows the same band without any routing to flip. The
attribution was removed rather than reworded.
