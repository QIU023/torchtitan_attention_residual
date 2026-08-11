# What this project can do overnight

Written 2026-08-11, after the items-2/3/6 batch. Ordered by value, with effort and the
gate each one needs. The scarce resource is the 8 GPUs: anything needing the full matrix
serialises against everything else needing it, so the plan interleaves GPU-bound gates with
work that is CPU-only or measurement-only.

## A. vLLM LoRA for `KimiLinearForCausalLM` -- do this first

The only thing between the adapter-only weight sync and an end-to-end verification.
Everything on our side is implemented and CPU-tested; the rollout refuses to load:

    ValueError: KimiLinearForCausalLM does not support LoRA yet

**Scope is smaller than the earlier note implied**, and I was wrong to file it as
"third-party, not ours" -- `/workspace/vllm_k3` is our own checkout, the same category as
the torchtitan fork. What a comparable model declares (`DeepseekV2ForCausalLM`, also
hybrid MoE + MLA) is two things: `SupportsLoRA` in the bases, and a
`packed_modules_mapping` for its fused projections.

Feasibility checked rather than assumed. `vllm/models/kimi_k3/nvidia/model.py` uses
9 `ReplicatedLinear`, 2 `RowParallelLinear`, 2 `ColumnParallelLinear`, and vLLM ships
`ReplicatedLinearWithLoRA`, `RowParallelLinearWithLoRA`, `ColumnParallelLinearWithLoRA`
for exactly those. `FusedMoEWithLoRA` exists too and is not needed, since we do not adapt
routed experts.

Two real unknowns, both with a known shape:

* **KDA must be excluded.** `KimiDeltaAttention` reads `linear.weight` directly for the fla
  kernels, so a LoRA wrapper there would be silently dead -- the same reason
  `apply_lora` skips the KDA subtree on our side. vLLM has `lora_skip_prefixes` for
  precisely this (it exists for MTP layers), so the fix is a declaration, not surgery.
* **Name agreement.** Our engine emits `...lora_A.weight` / `...lora_B.weight` plus
  `base_layer` on the wrapped bases. That has to match what vLLM's LoRA loader expects for
  these module paths. This is where the time will actually go.

Effort: half a night. Gate: the existing live/frozen digest protocol on
`grpo_k3_adapter_sync.sh`, which already runs both arms. Success criterion is sharp and
already scripted -- `adapter mode, base_sync_done=False` exactly once, then `True` every
step, with digests changing.

## B. CP/TP gradient equivalence -- the last real proof gap

Not a known bug. PP is proven bit-exact; CP and TP have only the absence of evidence plus
a strong indirect argument (removing KDA drops the deviation 67x, which points at the
kernel's shape sensitivity rather than the parallelism).

The instrument has to be a **shape-independent reference**: a naive recurrent KDA in
float32/float64, on one KDA layer and a short sequence. And it must be validated against
`chunk_kda` on a single rank BEFORE being used to judge cp2/tp2, or it is another untrusted
instrument -- this project has already spent days on exactly that mistake three times.

Effort: half a night, most of it in the reference and its validation. Gate: none, it adds a
probe and a document. Runs on 1-2 GPUs, so it can overlap a matrix.

## C. `#32` remainder: DEP knobs to config fields

`KIMI_VIT_DEP` and `KIMI_VIT_DEP_STAGES` are read from 8 sites across
`pipeline_adapter.py`'s PP layout builders, several with no config in reach. The four
vision knobs already landed because `encode_images` had `self.config`.

Effort: a few hours. Risk: real -- that file has produced two of my regressions this week.
Gate: full multimodal matrix. Keep the env override, as the vision knobs did, so the dozen
documents recording those commands stay correct.

## D. `#55` prerequisite: does the vision tower compile?

`#55` is not takeable as written (measured: `VarlenAttention` cannot be constructed here
without `flash_attn`; FlexAttention materialises the full `L x L` scores without
`torch.compile`, which every flavor here runs without). The overnight-able question is the
prerequisite: **turn `compile.enable` on for the vision tower and see whether it survives
the matrix.** If it does, FlexAttention becomes a real option and `#55` opens; if it does
not, `#55` is closed with a reason rather than a preference.

Effort: a few hours. Gate: full multimodal matrix.

## E. Full preprocessing parity against the released processor

`#62`'s judge covers the antialias question. The larger prize is unused: `/workspace/k3qat_mm_hf`
ships `kimi_k3_vision_processing.py`, `kimi_k3_processor.py` and `media_utils.py` -- the
actual release preprocessing. A parity test over the WHOLE pipeline (navit_resize
decisions, padding, normalisation order, patch order) against that reference would close a
class of silent mismatch rather than one setting, and the reference is already on disk.

Effort: half a night. Gate: none, it is a test. CPU-only, so it overlaps anything.

## F. `#12` remainder: 60 comment blocks

Mechanical but judgement-heavy: 60 blocks, ~870 lines, each a decision about whether the
explanation earns its space. The pattern that worked on the 46-line one is to move real
reasoning into the relevant PR body -- which improved the PR too, since it documented the
adapter's subtlest part where no reviewer would look.

Effort: a night on its own if done properly. Gate: AST-equality (comments only), which is
stronger and cheaper than a matrix run.

## G. PR26 and the PR slices -- prepared, not filed

PR26 (float32 grad norm) is verified on a clean upstream worktree and ready. Filing is an
outward-facing action and needs your go-ahead, so overnight work is preparation only:
rebase the slice onto current upstream, re-verify, and check the finding-11 rule (PR slices
must be fresh commits, never cherry-picks of the messages carrying
`pytorch/torchtitan#N` trailers).

Effort: a couple of hours. Gate: clean-upstream reproduction, already scripted.

## Not overnight-able here, and why

* **DEP's benefit.** `r = 25.2` at the debug flavor hides 0 of 32 encodes; the optimum near
  `r = 1` ceilings at 2.7% of a compute-bound step, and MFU here is ~0.09%. Arithmetic, not
  effort -- it needs different hardware, not another night.
* **flash-attn for SM 12.0.** A toolchain build with an uncertain outcome, and it would gate
  the vision path on a dependency nothing else in the model has.
* **`#13`** (rebuilding the AttnRes block stack). Hottest path, three documented
  subtleties in twenty lines, and the suggested buffer reuse is how gradients break
  silently. `#14` needs a tolerance argument first, which IS overnight-able as a written
  argument even though the change is not.

## Suggested ordering for one night

1. **A** (vLLM LoRA) with **E** (preprocessing parity) in parallel -- A is GPU-light until
   its final GRPO arms, E is CPU-only.
2. **B** (CP/TP reference) -- 1-2 GPUs, overlaps nothing else contending.
3. **C** or **D**, whichever you want gated first, then the single full multimodal matrix
   that covers it.
4. **F** and **G** as filler when a matrix is occupying the GPUs.

One matrix run per night is the realistic budget for code that touches parallelism, and it
belongs at the end so it covers everything.
