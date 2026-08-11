# Overnight: working the code-review findings

## Commit messages: cleaned (was a pre-push blocker)

Resolved. Every unpushed commit in both repos was rewritten so no bare `#N` remains: the
fork's twelve and the logbook's four. The reference-form scan comes back empty, `git diff`
against `backup/pre-msg-rewrite-2026-08-09` is empty in each repo, and all twelve fork tree
objects compare identical -- messages only. Findings now read `finding 49`, `findings 51/61`.

The hashes recorded below are post-rewrite. Already-pushed history was left alone on
purpose: its reference events have fired, a rewrite cannot unfire them, and it risks firing
them again.

## Traceback point (start of this session's overnight run)

    fork    QIU023/torchtitan        attention_residual_dev  3e7e23e3c0eb1085dd19fc53b3a5484ef8320eb4
    logbook QIU023/torchtitan_..._residual  main            d80b408100e6266b2d88ff641e9cec3b85c38080

Disk at start: 301 GB free (21% used) after clearing this session's test residue.
Watchdog is running as a supervisor service and prunes at 10 GB free.

## Rules for this run

* Anything that touches real functional code -- a behaviour change, a refactor, or a
  cleanup that moves code -- gets the **multimodal full-parallelism matrix** before it is
  called done. Reference values: `MATRIX_18_CORRECTED_2026-08-09.md` (post grad-norm fix).
* A finding that regresses the matrix, or that cannot be fixed without opening another
  line of work, is **recorded here and routed around** rather than left half-done.
* Numerical comparisons read TensorBoard, not stdout. stdout prints five significant
  figures, which is not enough -- CLAUDE.md says so, and this session spent hours before
  acting on it. `scripts/loss_compare.py` and the tb reader in the gate script do this.

## Carried in, unresolved: DEP forward divergence on the 30-layer flavor

> **RESOLVED 2026-08-10 -- see `DEP_30L_RESOLVED_2026-08-10.md`.** The divergence was a
> cold-init comparison: DEP moves the vision tower to another PP stage and torchtitan
> seeds PP ranks distinctly, so the two cold arms start from different weights (measured:
> same 880 parameters, global norm sum 7732.368 against 7733.152). From a shared
> checkpoint the arms are identical on the 30-layer flavor -- chunked and unchunked, pp2
> and pp4, including the tower split across two stages. Depth was never the variable, and
> the FSDP assertion below does not reproduce in five configurations. The section is kept
> as the measurement it was.

Parked deliberately, per the instruction to route around what does not resolve.

| configuration | DEP off vs on |
|---|---|
| 13-layer `report_arch`, seq 256 / 1024 / 4096 | **0.000e+00 at full precision, 3 steps** |
| 30-layer `report_arch_pp8vp4` + chunked(8), seq 256 | step1 **6.03e-04**, step2 1.48e-03 |
| 30-layer `bubble_ratio` + chunked(8), seq 4096 | step1 8.3e-03 |

What is established: the variable is **not sequence length** -- the 13-layer flavor is
exactly zero at every sequence length tested. It is something in the 30-layer flavor.

What is not established: whether chunked loss is the cause. The unchunked control could
not run:

    AssertionError: FSDP reduce-scatter expects uniform gradient dtype
                    but got {torch.float32, torch.bfloat16}

That is the same class of defect as the grad-norm one fixed earlier tonight -- mixed
gradient dtype -- but on a different collective, and it is loud here where the grad-norm
one was silent. Chasing it means opening a third line, so it is parked with the rest.

Why this matters for claims: **"DEP is numerically exact" holds for the 13-layer flavor
at full precision and does NOT hold at 6e-4 for the 30-layer one.** Every DEP gate run so
far used 13 layers; the 30-layer flavor appears only in pp8xvp4 tests where BOTH arms had
DEP on, so the offset cancelled and could not surface.

## Findings ledger

Status values: `fixed+matrix` (fixed and the matrix reran clean), `fixed` (fixed, matrix
not applicable -- docs, dead code), `parked` (recorded and routed around, with reason),
`n/a` (already fixed by this box's parallel commits, verified against HEAD).

Commits, in order: `0d7ce0f96` (nine findings), `b0f631f6a` (tier-1 checkpoint round trip),
`3e11c103b` (two defects an internal round trip cannot reach), `bf014239d` (block size).
Matrix runs `mxA`/`mxB`/`mxC` each came back **10/10 bit-identical** to
MATRIX_18_CORRECTED_2026-08-09; the three TP+CP cells fail as the documented
`KIMI_VIT_DYNAMIC_CP=0` limitation, unchanged.

| # | file | status | note |
|---|---|---|---|
| 51/61 | vit_prefetch.py | fixed+matrix | deferred side-stream join reverted to a join at issue time |
| 49 | multimodal_model.py | fixed+matrix | later vision share with sentinels but no grid_thw now raises |
| 50 | pipeline_adapter.py | fixed+matrix | n_vit>1 with zero wired vision stages now raises |
| 59 | moonvit.py | fixed+matrix | gather-KV key mask assumed padding was a trailing run; it is per frame |
| 19 | lora.py | fixed+matrix | NF4 branch dropped the base bias the other branches add |
| 20 | lora.py | fixed+matrix | `_forward_packed_tp` dropped it too; rowwise adds after redistribute |
| 43 | attn_res_model.py | fixed+matrix | `init_weights` walked `self.layers` only, missing nested MTP blocks |
| 48 | attn_res_model.py | fixed+matrix | MTP got the post-norm hidden state, applying hnorm twice |
| 40 | config_registry.py | fixed+matrix | MTP flavor set `global_vocab_size=163840` on a 2016-wide model |
| 37 | state_dict_adapter.py | fixed+matrix | `from_hf` never stripped the `language_model.` prefix that `to_hf` adds |
| 72 | state_dict_adapter.py | fixed+matrix | expert regex anchored the same way; same fix |
| 39 | state_dict_adapter.py | fixed+matrix | no K3-layout delegation, so latent MoE keys never mapped |
| 39b | state_dict_adapter.py | fixed+matrix | shared experts answered with the Kimi Linear layout under a K3 wrapper |
| 71/42 | hf_key_map.py | fixed+matrix | `g_proj` is one released name for two different gates |
| 24 | packed_mxfp4.py | fixed+matrix | E8M0 `0x00`/`0xFF` decoded against the OCP MX spec in both directions |
| 62 | vision_preprocess.py | **fixed+judged** | the judge exists after all: the release uses PIL bicubic, so a differential against it gives 240x/147x on high-frequency content and 1.0x on a smooth gradient (pinned as its own test) |
| 41 | attn_res_model.py (+4 files) | fixed+matrix | block size reached the model only through a lossy `num_blocks` trip |
| 21 | muon.py | fixed, matrix as control | weight decay 0.1 reached 1-D parameters through Muon's AdamW fallback |
| 22 | mxfp4_qat.py | fixed, matrix as control | the QAT wrapper had no `.weight`, so Per-Head Muon silently skipped every wrapped projection |
| 34/69 | parallelize.py | fixed+matrix | the dynamo carve-out discarded `torch.compiler.disable`'s return, so it carved nothing |
| 31/68 | pipeline_adapter.py | fixed+matrix | dead pipelining twin deriving `layers_per_block` by floor; deleted, 167 lines |
| 70 | pipeline_adapter.py | fixed+matrix | step-end cache sweep swallowed every exception with a bare `pass` |
| 67 | layout.py | fixed+matrix | layer->stage discovery is unreachable under multi-rank PP; now verifies the default instead |
| 65 | pipeline_adapter.py | fixed+matrix | the sweep evicted only what backward marked, so eval-only microbatches never left the cache |
| 66 | pipeline_adapter.py | partly fixed | a config error no longer degrades to a rank-local fallback; see below |
| 45/53/73 | model.py | fixed+matrix | every MoE parameter bucket matched nothing, so flops_per_token reported total rather than activated |
| 46 | moe.py | fixed+matrix | SiTU experts called `torch._grouped_mm` directly, bypassing the seam the MXFP8 converter overrides |
| 47 | model.py | fixed+matrix | base multimodal path kept the naive index assignment; both models now share one helper |
| 63 | multimodal_model.py | fixed+matrix | `num_images` was counted before CP sharding and compared after |
| 35 | multimodal_model.py | partly fixed | import-time method grafting, half of it dead; substantive claim disproven |
| 23 | lora.py | fixed+matrix | merge mixed a materialized base with sharded adapters |
| 64 | vision_preprocess.py | **not a defect** | the constraint holds structurally; see below |
| 74 | __init__.py | fixed+matrix | module-scope flavor discovery crashed the whole package import without fla-core |
| 18 | model.py | fixed+matrix | per-microbatch device-to-host sync on the embed path, now unconditional |
| 28 | model.py | premise wrong | `is_mla` is not constant-True; guard kept, error type corrected |
| 76 | experiments/kimi_k3 | fixed | orphan `__pycache__` directory removed (untracked) |
| 25/33 | pipeline_adapter.py | fixed+matrix | same as 31/68 -- the dead twin, deleted |
| 26/75 | model.py | fixed+matrix | the duplicate `dim`/`vocab_size` pair |
| 38 | model_configs.py | fixed+matrix | `is_k3_shaped` keyed on a name list that excluded the "mini" alias |
| 52 | test_vit_cp_plan.py | fixed | re-pinned t>1 with an uneven split -- the gap that let finding 59 exist |
| -- | distributed/utils.py | fixed+matrix | **core**: the total norm was computed in the gradients' dtype, so it depended on the PP/EP partition |
| 16 | quantile_balance.py | fixed+matrix | 92 serialised all-reduces per step, one per MoE layer, now one |
| 15 | moonvit.py | fixed+matrix | `cu_seqlens.tolist()` per block was 27 device-to-host syncs per tower forward |
| 13 | attn_res.py | **not attempted** | see below |
| 14 | muon.py | **not attempted** | see below |
| 50 | pipeline_adapter.py | **regression I caused, fixed** | the guard assumed every rank owns a vision stage once the tower splits; n_vit>1 could not start |
| -- | multimodal_model.py, moonvit.py | fixed+matrix | the split tower bypassed FSDP2's all-gather by calling forward_head/body/tail directly |
| 58 | model.py | fixed+matrix (numbers moved) | MLA's inner attention duplicated common's `ScaledDotProductAttention` |
| 57 | muon.py | fixed+matrix (**numbers moved**) | fallback AdamW was a clone of torch's; reuse costs a last-bit `exp_avg` difference that reaches 2.6e-03 by step 10 on the Muon flavor |
| 56 | packed_mxfp4.py | fixed+matrix | remaining half: decode delegates to torchao's `mx_formats.to_dtype`, bit-identical including E8M0 `0xFF` -> NaN |
| 29 | multimodal_model.py | fixed+matrix | LLaVA scaffold deleted, 552 lines with its test; takes finding 17's double `.item()` loop with it |
| 30 | config_registry.py | fixed+matrix | second `dataclasses` alias in one function |
| 27 | config_registry.py | fixed+matrix | stubs kept explicit; the missing agreement check against `SCALING_LAW_TABLE` now exists |
| 6/7 | model.py | **assessed, tables written** | two of their own rows were wrong: upstream broadcasts k_rot the same way, and the MLA's GQA fields were dead |
| 54 | parallelize.py | **fixed+matrix** | converged onto apply_fsdp_to_decoder, 182 lines -> 81. Read-only aliases, no FQN moves. First attempt deadlocked every multimodal CP cell (extra FSDP unit); fixed by aliasing on the wrapper, then 18/18 byte-identical |

### DEP, stated precisely, after two regressions of my own

Both found by actually running n_vit>1, which nothing had done since the guard landed:

* The finding-50 guard read `wired == 0 and n_vit > 1`, assuming every rank owns a vision
  stage once the tower is split. True only when n_vit equals the pipeline degree. At pp=4
  with n_vit=2 the two text-only ranks raised, so **n_vit>1 could not start at all**.
* The shares called `vision_tower.forward_head` / `forward_body` / `forward_tail` directly,
  and FSDP2 registers its all-gather on `__call__`, so `patch_embed.proj.weight` stayed a
  sharded DTensor and the conv failed. **The split tower had never run under FSDP**; every
  earlier n_vit>1 result was at dp_shard=1.

Where clause 2 now stands, at dp_shard=2 x pp=4 from a shared seed checkpoint:

| | step 1 | step 2 | step 3 |
|---|---|---|---|
| n_vit=1 | 11.81748 / 9.9896 | 11.78009 / 9.0480 | 11.30619 / 8.6702 |
| n_vit=2 | 11.81748 / 9.9896 | 11.78009 / 9.0480 | 11.30619 / 8.6701 |

Loss identical throughout; grad_norm differs in the last printed digit at step 3, which is
the limit of stdout's five significant figures. Not claimed as bit-identical.

**What can and cannot be said about bubble hiding.** The mechanism is implemented per the
report and is numerically exact: clause 1 (tower on its own stage) agrees with DEP off, and
clause 2 (tower split across stages) agrees with clause 1. Clause 3's placement is the
schedule's; our extra async run-ahead was implemented and then removed on measurement. The
BENEFIT is not demonstrated, and the hardware argument for why it cannot be here is
unchanged: r = 25.2 at the debug flavor gives 0 of 32 encodes hideable, r = 0.057 at real
scale hides 56% of an encode worth 0.6% of the step, the optimum near r = 1 ceilings at 2.7%
of a compute-bound step, and MFU here is ~0.09%. So: mechanism implemented and exact,
benefit unverified and unverifiable on this box.

### The core grad-norm defect, found by trying to align DEP

Written up in full in `DEP_ALIGNMENT_2026-08-09.md`, with the upstream kit in
`../Raising_PRs/PR26_torchtitan_grad_norm_low_precision/`. Short version:
`torch.nn.utils.get_total_norm` returns the norm in the gradients' dtype, so with
`training.dtype="bfloat16"` -- which this flavor sets deliberately, because fp32 KDA asks
108160 bytes of shared memory against this card's 101376 -- the reported norm is accurate to
three or four digits AND depends on how the parameters are grouped. Clipping fires every
step, so the error scales the update.

Two PP layouts with **identical gradients** reported 10.008054 and 9.951641; the float32
truth is 9.989287. With the norm computed in float32 both report 9.9893 and agree on every
step. The matrix references moved for this (step 1 unchanged everywhere, moves from step 2-5,
max abs delta 3e-4 to 3.4e-2) -- new table in `MATRIX_18_FP32NORM_2026-08-09.md`, and the new
numbers are the accurate ones.

### Two efficiency findings left alone, with the reason

`#13` (rebuild of the AttnRes block stack plus its float32 copy on every call) is the hottest
path in the model and carries three documented subtleties in twenty lines: the order of the
norm call is load-bearing for FSDP2's all-gather of `proj.weight`, float32 is load-bearing
because the zero-initialized pseudo-query makes its gradient a difference of nearly equal
terms (6x to 15x cancellation, measured), and under TP `proj.weight` is a replicated DTensor
the einsum has to mix with a plain tensor. The suggested fix reuses a buffer inside that
autograd graph, which is how gradients break without any symptom. It needs its own change
with gradient-level verification.

`#14` (batching the per-head Newton-Schulz) cannot be checked by the criterion used for
everything else here: batched and sequential matmuls need not agree bit-for-bit, so it needs
a tolerance argument before it can be verified at all.

`#38` is the most consequential of the late batch. `is_k3_shaped` decided the whole K3 delta
set (SiTU, the gates, the lower-bounded decay, q-compression, LatentMoE, the extra global
layer, block size 12) from `size in ("2p8t", "k3mini")`. The rename
`kimi_linear_k3mini_* -> kimi_k3_mini_*` made the parsed size "mini", which is aliased to the
same row -- but not to the name list. So since that rename, every flavor built through
`model_registry` under the new name has had silu rather than SiTU and block size 3 rather
than 12, i.e. 7 equal blocks instead of the 2-with-a-9-layer-tail shape that is the entire
reason the row exists. The trainer config of the same name builds it correctly and says so in
its docstring. Two spellings of one row disagreeing on structure is the seed-checkpoint
mismatch the finding named.

Still open after this: the trainer config function and the registry flavor share the name
`kimi_k3_mini_block_attn_res`, so a consumer resolving that name gets full extents (163840
vocab, 32 experts) where the trainer builds the documented debug extents (2016, 8). Fixing
that is a rename touching launch scripts and logbook references -- a decision, not a
cleanup, so it is recorded rather than taken.


`#74` is the one worth reading twice, because the fix restores a mechanism rather than adding
one. The package already guards the fla-core import: a `try` around the re-export block
records `_KIMI_IMPORT_ERROR` so that a CPU-only box can import the package and get a pointed
message only when a flavor is actually built. That deferred raise was unreachable, because
`kimi_k3_configs` is built at module scope AFTER the guard and walks
`config_registry -> model_configs -> model`, whose fla import raises outside any `try`.
Verified both directions by blocking `fla` with a meta-path finder: before, `import
torchtitan.models.kimi_k3` died; after, it imports, warns that the flavor list is empty, and
`model_registry` still raises "Kimi K3 flavors require fla-core". Flavor count with fla
present is 22 either way.

`#28`'s premise does not hold. The finding calls `is_mla` constant-True by construction and
asks for the dispatch to be flattened and the `NotImplementedError` deleted. But a config
with all five MLA dims None and `mla_use_nope` False constructs fine, so the branch is
reachable and deleting it would remove a live guard. What was wrong is the exception type: an
unimplementable configuration is a user error, so it is now a `ValueError` with a message
saying which fields to set.

### Two findings that turned out not to be defects, and how that was established

`#64` (pack_video records only frame 0's resolution) reads like a real hole, and I wrote the
validation before checking whether it could fire. It cannot. `prepare_image` derives its
grid from `navit_resize(W, H, ...)`, a pure function of the input dimensions and the kwargs,
and `pack_video` takes a single stacked `[F, C, H, W]` tensor -- so every frame shares H and
W and necessarily resizes to the same grid. The finding assumed `pack_video` accepts ragged
frames the way `pack_images` does; it does not. So the check I had added was dead code, which
CLAUDE.md rules out ("no speculative defensive checks"), and I reverted it. What landed
instead is a docstring saying WHY the constraint holds, so the next reader does not re-raise
it.

`#35`'s substantive half is also absent. The claim is that grafted `init_weights` skips the
MoonViT tower, leaving it at `torch.empty` garbage for from-scratch VL training. But
`KimiK3MultimodalModel` defines its own `init_weights` that does init the tower, and the
graft loop is guarded by `if not hasattr(...)`, so the `init_weights` entry never fires.
Measured rather than argued: two same-seed builds of the multimodal flavor agree on all 30
tower/projector parameters and every value is finite -- the same test that caught the real
version of this in finding 43. What IS true is the code-quality half: attaching
`get_attention_masks` to the class at import time is invisible to anyone reading the class,
and half the loop was dead. That method is now written out explicitly and the loop is gone.
The remaining external assignments (`layers`, `verify_module_protocol`) are the same smell
and are left to the cleanup pass rather than churned here.

`#63` is a third case worth distinguishing from both: the defect is real as code (a count
taken in one scope and compared in another) but not reachable as a failure. Under CP the
shard is sized by this rank's own sentinel count, so post-shard rows always equal
`num_sentinels` and the per-token branch necessarily wins; the one convention where they
could differ is already rejected inside `_select_cp_shard`. Fixed anyway, because making the
comparison well-scoped costs nothing -- but recorded as unreachable rather than as a bug
that was biting.

### #45/53/73: the MFU numbers, and which of them need a correction

All four bucket patterns in `get_nparams_and_flops` got **zero** hits. The real names are
`ffn._moe.routed_experts.inner_experts.*`, `ffn._moe.router.gate.weight` and
`ffn._moe.shared_experts.*`; the code looked for `.moe.experts`, `.moe.router` and
`.moe.shared_experts` -- `_moe` with a leading underscore, and `routed_experts` rather than
`experts`. Every MoE parameter fell through to `dense`, i.e. was counted as activated.

The fix has an external check, which is unusual for this batch and worth stating: with the
patterns corrected, the K3-shaped `2p8t` flavor reports **104.19B** activated linear
parameters against K3's published **104.2B activated**. The broken version reported 2.78T
-- exactly the model's TOTAL. It was publishing total-as-activated.

Measured inflation of `flops_per_token`, which is what MFU divides:

| flavor | broken | fixed | overstated |
|---|---|---|---|
| `report_arch` (the matrix flavor) | 0.4806 G | 0.4063 G | 1.18x |
| `bubble_ratio` (the DEP latency runs) | 0.8044 G | 0.6248 G | **1.29x** |
| `debugmodel_block_attn_res` | 0.083 G | 0.041 G | 2.02x |
| `2p8t` (896 experts, top_k 16) | 16689 G | 645 G | **25.89x** |

**Every MFU and TFLOPs figure this box has printed is too high, by a factor that depends on
the flavor's expert count.** My first pass at this annotation said "about 2x" everywhere,
which was wrong: 2.02x is the `debugmodel_block_attn_res` figure, and the runs that produced
the widely-cited `mfu 0.12%` are `bubble_ratio`, where the factor is 1.29x. So that reads
**~0.093%**, not ~0.06%. Confirmed live rather than only on paper: the matrix's `dp1` cell
went `tflops 0.71 / mfu 0.23%` to `tflops 0.60 / mfu 0.19%` across the fix, matching the
1.18x predicted for that flavor.

The direction matters for what was concluded: a lower MFU means the GPU is idler still, so
"the ViT encode is already free, a run-ahead cannot help" gets stronger. That document's
arithmetic -- 0.12% x 5.2% x 50% ~ 0.003% -- becomes ~0.0023%. Same conclusion, smaller
number, and no conclusion there depended on MFU being as high as 0.12%. Both
VIT_DEP_DESIGN_2026-08-07 and HANDOFF_2026-08-08_EVENING now carry the correction inline.

Why the existing test missed it: `test_nparams_and_flops` asserted `1e6 < flops < 1e11`, a
band spanning five orders of magnitude, which a 26x error passes comfortably. That
assertion is now a narrow band, and the new test pins the 2p8t flavor against the released
activated count.

**Two of these interact, and the second exists because of the first.** `#67`'s new
guard inspects the layers THIS rank holds, so unlike every other bail-out in the install
path it can fail on one rank and not its peers. The broad `except Exception` around the
layout build would then turn that into "this rank runs without the adapter" -- and a rank
with no adapter sends no delta, so the first cross-stage hop hangs instead of reporting the
real problem. `#66` is therefore closed only to the extent that matters: a `ValueError`
from the layout builder now propagates. The general concern it names -- that the whole
install decision is made from rank-local state -- is not closed. Every other path reaches
the same answer on every rank because the attributes come from the same model class, so a
setup-time all-reduce for unanimity would be defensive against a case I cannot construct.
Recorded rather than implemented, with the reasoning, so it can be revisited if a real
divergence turns up. `#66` has no unit test for the same reason: the failure it prevents
needs a multi-rank non-contiguous split.

`#34/#69` deserves a note on why it survived so long. The comment above the broken loop
explains the carve-out correctly and at length, and twenty lines below it the same file
does the equivalent patch for `block_attn_res` the RIGHT way, including a comment
explaining that each `from X import Y` needs its own rebinding. The knowledge was present;
only the three-line loop was wrong. It survived because nothing could observe it, which is
why the fix pulls the carve-out into its own function -- observability was the actual
defect.

`#31/#68` leaves a naming inconsistency worth carrying into the PR text: the RFC and
several phase docs name the deleted `pipeline_llm_with_cache_adapter` as the entry point.
That was accurate when written. The live entry point is
`pipeline_kimi_k3_with_cache_adapter`; archived RFC drafts stay as historical record.
| -- | model.py | fixed+matrix | duplicate `dim`/`vocab_size` properties in one class (flake8 F811) |
| -- | distributed/utils.py | fixed+matrix | **core** defect: grad-norm dtype mismatch across `pp_mesh` |
| 60 | multimodal_model.py | see below | `_select_cp_shard` added to the DEP path |

`#62` is marked weakly judged deliberately. Antialiasing cannot be detected by any test
this repo can write on its own -- it needs an external reference to compare against, so
what landed is a no-regression guard plus the code fix. Its real judge is the reproducible
HF-reference parity test the review asks for separately; that test is now a stated
prerequisite rather than a nice-to-have. `#24` is in the same family (E8M0 `0x00`/`0xFF`
are codes our own quantizer never emits, so the module's round trip structurally could not
reach them) but is verifiable, because a test can feed the decoder those codes directly.

`#41` scope was enumerated rather than assumed: over every registered size, k3mini is the
only flavor whose partition changes (11+10 -> 12+9). The 93-layer `2p8t` flavor was already
correct, both debug flavors use block size 1, and the matrix flavors pass `num_blocks`
directly -- which is why the matrix is expected bit-identical here, not merely convergent.

### The matrix cannot judge #21 or #22, and that had to be checked rather than assumed

Both were queued as "changes numbers, so run the matrix". Inspecting the matrix flavor first
showed why that plan was wrong: `kimi_k3_debugmodel_report_arch` configures a single AdamW
group with pattern `.*` and `weight_decay=0.1`, so it never touches `default_muon()`. QAT is
likewise off, and the scope that reaches MLA projections (`all_linear`) is an ablation knob
the matrix does not select. Running the matrix against these two changes therefore proves
only that nothing leaked into the default path -- worth having, but it is a control, not a
judge. Recorded because "the matrix passed" would otherwise read as verification it cannot
provide.

The real judges:

* **#21** -- a 10-step A/B on `kimi_k3_mini_muon` (2 GPUs, FSDP; single GPU is unavailable
  here because the fla KDA kernel asks 108160 bytes of shared memory against the card's
  101376). Steps 1 and 2 agree to all printed digits, step 3 onward diverges by 1e-4 to
  4e-3, and both arms converge alike (4.70299 vs 4.70680 at step 10). That magnitude is the
  prediction, not a surprise: `weight_decay=0.1` at `lr=3e-4` shrinks a parameter by 3e-5
  per step, so the first visible digit should move around step 3.
* **#22** -- tag counts, which state the defect better than a pass/fail does. On the
  4-layer dense-MLA model: 8 projections tagged without QAT, 24 wrapped by QAT, and **0**
  tagged with it. After the fix, 8 again. Per-Head Muon was not degraded, it was gone.

One correction to how I first described #22's severity. I wrote that the existing
"per-head is inert" warning would stay silent because unwrapped projections still tag.
That is wrong for an all-MLA model -- every per-head target gets wrapped, nothing tags, and
the warning does fire. It is right for a mixed model, which is the realistic case, and that
is now measured rather than argued: on the debug flavor (KDA layers 1-3, full attention on
layer 4) tags go 11 -> 9 under QAT, because KDA projections are deliberately never wrapped.
Nine is greater than zero, so the warning stays quiet while a third of the tags are gone.

### Routed around, with reasons

| # | what | why routed around |
|---|---|---|
| 44 | MTP + chunked loss | needs an `mtp_loss` redesign, not a local fix |
| 32 | env-var topology | upstream will not accept env vars; migrating to config fields is its own change |
| 45/53/73 | MoE FQN patterns never match | inflates MFU/TFLOPs 10-30x; needs a decision on whether any published MFU number gets retracted |
| -- | TP x dynamic CP defect 2 | gather-KV on DTensors needs the `wo` local/DTensor contract settled first |
| -- | DEP 30-layer 6e-4 divergence | RESOLVED 2026-08-10, a cold-init comparison; see `DEP_30L_RESOLVED_2026-08-10.md` |
| -- | 35 flake8 findings in this folder | 27 F401 + 7 F811 + 1 E301, 36 of them in `config_registry.py`. All unused or shadowed imports, none functional. That file is under concurrent cleanup, so fixing them here would only collide -- but they do fail CI lint and have to land from one side or the other. |

The one lint finding not routed around is the F811 pair in `model.py`: two identical
`dim`/`vocab_size` property definitions in a single class, where the later pair silently
shadowed the earlier. No behavior change, but it is duplication rather than style, and
`model.py` is not the file under concurrent rewrite.

**#60 detail, because its status is genuinely ambiguous.** The DEP path called
`_exchange_sentinel_counts` and discarded the result, never calling `_select_cp_shard`
that the non-DEP path uses. The call is now added, which makes the two paths structurally
identical -- but the numbers did not move on any configuration tried, so **the fix is not
supported by evidence**, only by the code reading. It stays in because it removes a real
divergence between two paths that should agree; it is not claimed as verified. A test that
actually exercises a non-identity shard selection is still missing.

> **UPDATED 2026-08-10 -- `DEP_60_VERIFIED_2026-08-10.md`.** The fix now has evidence, and
> the sentence above is wrong about why. Every configuration tried here had a degenerate
> partition (`counts=[64, 0]`), where selecting and not selecting give the same tensor:
> rank 0's slice is the whole encode and rank 1 has nowhere to splice. Forcing a split
> partition -- seq 96, so the 48 visual tokens straddle the midpoint, `counts=[47, 1]` --
> makes the pre-fix path fail outright rather than drift. The missing non-identity test now
> exists as 6 CPU tests in `test_cp_shard_selection.py`. The code comment's separate claim
> of a 0.042 divergence does not reproduce and has been withdrawn.
