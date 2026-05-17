# Fork rebase task — torchtitan + sglang sync to upstream/main

> Discovered 2026-05-17 while preparing the upstream PR batch. Not a PR
> itself — this is internal fork maintenance that **must happen before**
> the next GRPO run on a freshly-pulled fork.

## Why this is needed

Both forks have drifted from upstream:

| Fork | Ahead | Behind | Last sync |
|---|---|---|---|
| `QIU023/torchtitan` (`5cc52d0b`) | +208 | -71 | ~2026-04-12 |
| `QIU023/sglang` (`dc154e785`) | +450 | -20 | ~2026-05-10 |

Upstream torchtitan landed `627f4a31 [rl] Trainer refactor (#2985)` on
2026-04-20, which fundamentally restructured `PolicyTrainer.__init__`
and `_build_model`. The fork's `phase11_rlhf_grpo_infra/rlhf/run_grpo_*.py`
launchers will break in 4 places after merging upstream:

## Required reconciliation per file

### 1. `torchtitan/torchtitan/experiments/rl/actors/trainer.py`

Structural merge needed. Upstream's `_build_model` only returns a
random-init model; checkpoint loading was moved out to `CheckpointManager`
in commit `8cdfdc23 [rl] Better initial weight loading`.

Fork-side hooks that need re-anchoring:
- `_load_initial_dcp_weights` (fork-only)
- `_load_initial_hf_weights` (fork wrapper around HF safetensors)
- VLM injection: `_vision_tower`, `_projector`, `image_token_id` (used inside
  `compute_token_log_probs`)
- `_transfer_dtype` (fork name; upstream renamed to `_generator_dtype`)
- `kl_coef` (fork-added field; upstream pushes KL into `config.loss.build()`)

Decision needed: **adopt upstream's CheckpointManager path** (preferred,
keeps us close to upstream) **vs keep fork's bespoke load functions**.
The CheckpointManager path requires writing a small adapter for our
DCP-only ckpts (`hf_step3100`, AttnRes 447M) since those don't have
HF state_dict_adapters.

### 2. `phase11_rlhf_grpo_infra/rlhf/run_grpo_kimi_attn_res.py` (and 3 siblings)

Three concrete edits per launcher:

a. **`PolicyTrainer` ctor call site** (around line 122):
   - Pass `compile_config=...` (new required kwarg)
   - Pass `output_dir=...` (new required kwarg)
   - Rename `transfer_dtype=` → `generator_dtype=`
   - Drop any fork-only kwargs the trainer no longer accepts after
     reconciliation (depends on decision in #1 above)

b. **Delete the `parallelize_fn` wrapper** (around lines 258-285):
   ```python
   # DELETE THIS WHOLE BLOCK after upstream merge — trainer.py:243
   # now passes training/ac_config/dump_folder itself.
   def _patched_parallelize(model, *, parallel_dims, parallelism, compile_config, ...):
       ...
   model_spec.parallelize_fn = _patched_parallelize
   ```

c. **Audit `SGLangGenerator` ctor kwargs** vs upstream `VLLMGenerator.__init__`
   to keep the polymorphic spawn call shape valid. (Probably zero changes —
   the launcher only passes `config.generator, model_spec=, model_path=`,
   which are stable.)

Affected launchers:
- `run_grpo_kimi_attn_res.py`
- `run_grpo_llava_kimi.py`
- `run_grpo_llava_caption.py`
- `run_grpo_sum_digits.py`

### 3. sglang fork — lighter merge

Only 20 upstream commits to absorb; 8 touch `tokenizer_manager.py` (we
already tested cherry-pick of the env-gate patch into upstream successfully,
so the merge direction will also be clean).

After merge, our SHM-MM env-gate hunk at line 2726 will move to wherever
upstream's `824ad2414 Convert local-only self.X attributes to locals`
relocated `_determine_tensor_transport_mode` to — should be the same file,
different line number.

## What is NOT affected

- `SGLangGenerator` ([torchtitan/experiments/rl/actors/sglang_generator.py](../torchtitan/torchtitan/experiments/rl/actors/sglang_generator.py))
  — purely additive, upstream has no file by this name, zero conflict
- `eager_generator.py`, `grader.py` — also additive
- `experiments/rl/models/parallelize.py:parallelize_qwen3` (narrow signature
  fork still uses) — only called from `vllm_wrapper.py:214`, not from the
  trainer path
- Algorithm-stage Phase 2/3/4 work (torchtitan model definitions, AttnRes
  block decoder) — those files unchanged upstream

## Recommended ordering

1. **First**: file PR #1, #7 from the locally-prepared branches. Those are
   independent of fork rebase and credibility-building.
2. **Then**: do this fork rebase as one focused PR-shaped change inside the
   fork (not upstream). Aim for one commit per launcher + one commit for
   trainer.py.
3. **Then**: run the GRPO smoke pipeline on cloud GPU end-to-end to
   confirm post-rebase fork still produces coherent rollouts (1h smoke
   from `run_grpo_smoke_1h.sh`).
4. **Then**: resume the rest of the upstream PR batch (#3, #8, #9, #11).

## Effort estimate

| Sub-task | Effort |
|---|---|
| torchtitan trainer.py merge + VLM hook re-anchor | 0.5-1 day |
| 4 launchers ctor + wrapper cleanup | 2-3 hr |
| sglang merge (mostly mechanical) | 1 hr |
| GRPO smoke validation on cloud GPU | 2 hr including queue |
| **Total** | **1-2 days** |
