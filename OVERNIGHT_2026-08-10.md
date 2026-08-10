# Overnight, 2026-08-10

Starting HEADs: fork `b66280b1b`, logbook `5148fc3`, verl `1059a888`. Disk 257 GB free.

Plan was five phases; what actually happened is below, including two items deliberately
routed around and one blocked from outside. The logbook branch had diverged mid-run -- a
commit landed on the remote that was not mine -- so these sit rebased on top of it rather
than force-pushed over it.

## Done

### 1. veRL adapter-only weight sync -- the item queued longest

`peft_config` is now produced, so `engine_workers`' base-then-adapter sequence is reachable.
It had been returning `None` unconditionally behind a `# TODO: support Torchtitan PEFT`,
which meant a run asking for `model.lora.merge=False` still received merged full weights --
LoRA bought optimizer and gradient memory but none of its sync bandwidth.

Shape of it: wrappers are discovered from the module (not from config -- `apply_lora`
matches leaf names AND qualified suffixes and skips the KDA subtree, so a config-derived
target list would name targets that were never wrapped); rank and alpha are read off a
wrapper; the base half is the FULL model with `base_layer` inserted only on wrapped
projections; the adapter half is `lora_A`/`lora_B`, UNSCALED, because PEFT re-applies
`lora_alpha / r`.

Two decisions worth keeping:

* **Adapter mode is opt-in.** `model_config.lora` defaults to an EMPTY dict, so reading
  `merge` off it with a `False` default -- which is what the megatron engine does -- would
  have flipped every existing LoRA run onto the new path. The merged path is the one with
  end-to-end evidence, so an empty block stays merged.
* Names come from the state-dict adapter's own `_tt_key_to_hf`, with `_is_text_only` asked
  once against the whole dict. Calling `to_hf` per key would let a one-entry dict
  misclassify text-vs-multimodal and emit the wrong prefix.

**Verified:** 8 CPU tests; the merged arm of a 3-step GRPO run completes with four distinct
sync digests, so the existing path is unregressed.

**NOT verified end to end, and the blocker is external:** vLLM refuses the rollout with
`ValueError: KimiLinearForCausalLM does not support LoRA yet` at model load, before any
training step, so the adapter half never gets a sync to perform (0 adapter-mode log lines,
consistent with that). Making that architecture implement `SupportsLoRA` in the rollout
engine is the next step and is a change to the vLLM fork, not to ours.

Three of my own mistakes cost two GRPO launches: I pointed a multimodal flavor at
text-only weights (`Missing key: language_model.lm_head.weight`), then at multimodal
weights whose MLA dimensions vLLM rejects (`No valid MLA prefill backend found`), before
using the text stack every working GRPO script here already used. The flavor/weights/rollout
triple has to agree and I checked it last.

### 2. Findings 6, 7 and 54 -- the difference tables, and a split nobody had seen

`phase13.../REUSE_DIFF_TABLES_2026-08-10.md`. The maintainer asked two of these close to
verbatim. Neither answer needs a defence of forking; both resolve to "converge, in this
order, at this cost".

Two corrections to the findings themselves:

* **Finding 6's "broadcast k_rot across heads" is not a difference.** Upstream does the same
  `expand`. A PR body citing it would have been wrong in front of the maintainer who wrote
  that code.
* **New finding:** the MLA read `num_key_value_heads` / `num_key_value_groups` and never used
  them -- MLA shares one latent KV by construction and upstream's MLA has no GQA fields at
  all. Deleted.

The actionable output: **the three reuse items split rather than sharing one gate.** 54 is
unblocked (upstream's `apply_fsdp_to_decoder` only READS what it needs -- no assignment to
any model attribute anywhere in the function -- so three read-only properties satisfy it and
properties are not in `_modules`, so no FQN and no checkpoint key changes). 6 and 7 do share
one decision: adopt upstream's parameter names and migrate the DCP keys, or keep the
official-export names and justify on that basis.

### 3. Reuse and cleanup landed

* **finding 57**, Muon's fallback AdamW now calls `torch.optim.adamw`. Checked BEFORE
  replacing: 5 float32 steps give bit-identical parameters and identical `exp_avg_sq`, but
  `exp_avg` differs in the last bit because torch fuses the first moment as a `lerp`. That
  compounds -- measured on `kimi_k3_mini_muon`, 10 steps: identical for two steps, then
  2.6e-03 relative by step 10, same convergence. **This is a declared numerics change and
  prior Muon runs are no longer bit-reproducible.** Kept because torch's is the reference
  implementation, but the numbers are here to reverse it if bit-continuity matters more.
* **finding 56** (remaining half), MXFP4 decode delegates to torchao's `mx_formats.to_dtype`.
  Bit-identical, and checked on the cases that can distinguish the two rather than random
  data alone: three shapes at bf16 and fp32, plus all three E8M0 special values -- `0x00`,
  `0x7F`, and `0xFF` as NaN. That last one is a fix this function already carried and
  `quantize_mxfp4` never emits it, so a random-scale comparison passes either way. fp16
  targets are NOT equivalent (E8M0 scales reach 2**23) and the docstring says so. DCP's own
  `_dequantize_tensor_mxfp4` was the other candidate and is unusable -- a private method
  bound to `ReadItem` slicing and a 4-D layout.
* **finding 29**, the LLaVA scaffold is gone: 552 lines including its test. It described a
  frozen tower plus a separate projector, the opposite of the release recipe (report 2.4
  trains MoonViT-V2 jointly, projector is a tower child). Removes finding 17's double
  `.item()` loop with it.
* **finding 30**, one function carried a second `dataclasses` alias beside the one that has
  since gone.
* **finding 27**, the twelve sweep stubs stay explicit -- registries are read with `getattr`,
  and a module-level `def` is greppable and appears in tracebacks where an injected global
  does not. What the finding was right about is that nothing checked them against
  `SCALING_LAW_TABLE`, so `test_scaling_law_stubs.py` does. Its first version asserted
  "table cross product == stub set" and failed on three real exceptions (`2p8t` under a
  different prefix, `447m_aligned` only with `_n4`, `528m_l16` with no row), so the
  invariant is now the narrower true one with the exceptions named.

### 4. Findings 4 and 5

Three comments pointed at files only this repo has (two probe scripts, a local copy of the
released config); zero `phase13` references remain in the folder. `kcp.py`'s halo history
goes from eight lines to two, and the full story moves to the PR-D body -- which is where
writing it caught that body still claiming "`kcp.py` implements the halo exchange", untrue
since the hand-rolled version was replaced by fla's `causal_conv1d_cp`. Filing PR-D as
written would have described code that no longer exists, to reviewers who would open the
file.

No matrix for that one, on a stronger argument than a matrix: with docstrings stripped, the
executable AST of all three files is IDENTICAL before and after, so nothing but comments
changed.

## Routed around, with the reason

* **finding 54 implementation.** Unblocked and short, but it moves FSDP grouping on all
  eighteen cells, which moves the matrix baseline re-established this session and makes
  "different grouping" hard to separate from "regression" in one run. That is a deliberate
  call, not an end-of-overnight one.
* **finding 12** (48 comment blocks of 8+ lines). Untouched, and not because it is hard:
  it is 48 judgement calls about which explanations earn their space, in a codebase whose
  comments carry reasons that cost real time to learn. That wants reviewing with someone,
  not sweeping at the end of an overnight. Findings 4 and 5 were the precise part of the
  same tier and are done above.
* **finding 36** (magic suffix parsing to structured fields). It has already hidden 37
  flavors once, and it is the code path `VERL_TORCHTITAN_FLAVOR` resolution depends on --
  which tonight's veRL work leaned on heavily. Not worth changing in the same session.
* **finding 32** (env-var topology to config fields), unchanged from the earlier assessment:
  seven variables, about thirty consumers including recorded repro lines in a dozen
  documents.

## Matrix: 36 of 36 cells byte-identical

Both arms, DEP and dynamic CP on for multimodal, 10 steps, eager:

| arm | cells | result |
|---|---|---|
| multimodal `report_arch` | 13 + 5 max-degree | **18/18 pass, byte-identical** to the post-defect-2 table |
| text `kimi_k3_mini_block_attn_res` | 13 + 5 max-degree | **18/18 pass, byte-identical** to tonight's text table |

That is the result tonight's changes should produce: five of them touch code
(dead MLA fields, the alias rename, the AdamW reuse, the MXFP4 delegation, the 552-line
scaffold deletion) and none of them is on the matrix flavor's numerical path -- the AdamW
change only affects the Muon flavor, where it is measured separately at 2.6e-03 over 10
steps.

`collect13.sh` output for both arms is committed beside this file as `mm_13.txt` /
`text_13.txt`.

## State

| repo | HEAD | pushed |
|---|---|---|
| fork `attention_residual_dev` | `ec1192b2f` | yes |
| logbook `main` | this commit | yes |
| `verl` `kimi_k3_integration` | `ac913940` | yes |

Tests: 353 passed, 1 skipped, 169 subtests in the kimi_k3 folder; 11 in verl's two engine
test files. Lint clean on the CI-enforced codes.

## Next, in the order I would take it

1. **`SupportsLoRA` for `KimiLinearForCausalLM` in the vLLM fork.** It is the only thing
   between the adapter-only sync and an end-to-end verification, and everything on our side
   is already in place and CPU-tested.
2. **Finding 54.** Unblocked, short, and the highest-value remaining reuse item. Needs one
   matrix run because FSDP grouping moves, and needs the decision to accept that the
   baseline moves with it.
3. **Findings 6 and 7's shared decision**: adopt upstream's parameter names and migrate the
   DCP keys, or keep the official-export names and justify the fork on that basis. One
   decision, both components.
4. **CP/TP gradient equivalence.** Still a proof gap rather than a known bug -- the naive
   recurrent KDA reference is the way to close it, and it has to be validated against
   `chunk_kda` on one rank first or it is another untrusted instrument.
5. File PR26, and `#55` / `#36` / `#12` when there is appetite for the churn.

## Method notes worth keeping

* **Check the instrument before believing it, twice over tonight.** The AdamW and MXFP4
  swaps were both verified BEFORE replacing the code, which is the only order in which
  "identical" can be established -- and the MXFP4 check only meant something because it
  included the E8M0 special values that random scales never reach.
* **A test asserting a false invariant fails loudly and looks like a bug.** The scaling-law
  stub test's first version reported three "extra" flavors that are deliberate exceptions.
  The failure was mine, not the registry's.
* **`collect13.sh` on a live directory reports in-flight cells as `FAIL (n/10)`.** Two cells
  were misread that way before the arm finished. Collect once, after DONE.
* **Do not edit source while a matrix is running** -- a later cell imports the edited file
  and the run silently covers mixed code. That is why findings 4/5 waited.
* **The flavor, the weights and the rollout engine all have to agree**, and I checked that
  last: two GRPO launches died on a multimodal flavor against text weights, then on
  multimodal weights whose MLA dimensions vLLM rejects.
