# veRL fork -> upstream PR split, first cut (2026-09-17)

Repo `/tmp/wt_verl_0915`, base `upstream/main` = `67858929`, head = `549c1e21`
(`git diff upstream/main HEAD` = 27 files, +2817 / -92). First cut was on head `31079117`; regenerated on `549c1e21`, see the last section.
Plan followed: `Raising_PRs/PR_VERL_K3/PR_SPLIT_PLAN_2026-09-16.md`.

## Application order (this is the documented order)

    upstream/main
      -> 01_engine_tp_packed_and_compat.patch
      -> 02_engine_pipeline.patch
      -> 03_engine_ep_lora_qat_sync.patch
      -> 04_engine_context_parallel.patch
      -> 05_kimi_k3.patch
      -> 06_metrics_logprob_diff.patch
      -> local_env_and_diagnostics.patch      <-- patch 0 is applied LAST
      == HEAD

Each patch is a `git diff` between two consecutive trees, so each applies with
`git apply -p1` on the result of the previous ones and on nothing else.
Patch 0 last (rather than first) is what keeps 01-06 free of the local hacks: the
BFX9 wrapper, the diagnostics and `partial_dtensor` are added on top of the clean
upstream stack, exactly as the branch would carry them as a local patch.

## How it was produced

`work/splitlib.py` + `work/assign.py` + `work/build.py`. The split is at LINE
level, not hunk level: every added and removed line of the full diff is assigned
one ordinal, a line may carry a different text at different ordinals (a variant),
and a few line blocks are reordered so that every intermediate tree is valid
Python. Intermediate trees are built in a scratch index and the patches are
`git diff tree_k tree_k+1`. `work/` also holds the two staged variants of
`tests/workers/test_torchtitan_engine_cp_config.py` (`cp_config_01.py`,
`cp_config_04.py`) and the verification transcript (`work/verify.txt`).

Checks run on every intermediate tree, not only on the final one:
`ast.parse` on every .py file at every stage (23 files x 7 stages, 0 failures)
and `ruff --select F821,F811,F822` (undefined / redefined names) per stage,
all clean.

---

## 01_engine_tp_packed_and_compat.patch  (+461 / -28, 25 hunks, 8 files)

    tests/workers/test_torchtitan_engine_cp_config.py            +115/-0    1 hunk (new, staged version)
    tests/workers/test_torchtitan_engine_receiver_names.py        +61/-0    1 hunk (new)
    verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml     +9/-0    3 hunks
    verl/trainer/config/engine/torchtitan.yaml                     +9/-0    1 hunk
    verl/trainer/config/ref/torchtitan_ref.yaml                    +3/-0    1 hunk
    verl/workers/config/engine.py                                  +9/-0    2 hunks
    verl/workers/engine/torchtitan/transformer_impl.py           +235/-26  12 hunks
    verl/workers/engine/torchtitan/utils.py                       +20/-2    4 hunks

Carries, in `transformer_impl.py`: `import inspect` / `import math`; the
`components.checkpoint -> components.checkpointer` and
`components.lr_scheduler -> components.optimizer` import moves; the
`verl.utils.ulysses` import (`gather_outputs_and_unpad`, `ulysses_pad`);
`_parallelism_compat_kwargs`; the `config_registry` function-flavor fallback in
`__init__` (with the `model_registry` first attempt); the untied-checkpoint
`enable_weight_tying = False` override; `enable_sequence_parallel=`;
`**_parallelism_compat_kwargs(...)` replacing `spmd_backend=`; `initial_load_path`
(`dcp_path`, `initial_load_in_hf`, `initial_load_path`) and the checkpoint
`interval=1`; the `ParallelDims`-from-trainer change in `_init_device_mesh`; the
`_get_data_parallel_mesh` candidate list (dp mesh without an fsdp mesh); the
folded-stream probes `_folded_token_stream` / `_forward_params` /
`_forward_takes_cu_seqlens`; the folded branch of `model_forward_step`; the
extraction of `_finish_pred` and its TP (vocab-sharded and `_sp_group`) gather;
`_MULTIMODAL_KEYS`, `_squeeze_folded`, `_cu_seqlens_from_positions`; in
`prepare_model_inputs` `cp_pad_len = 0`, `pad_multiple = 1`, the TP `math.lcm`
padding, the `ulysses_pad` call and the whole `if cp_pad_len:` renumber /
label-pad / mask-rebuild block, the `if self._folded_token_stream:` branch
(`cu_seqlens` + the model's own `get_attention_masks`), `output_args["cp_pad_len"]`;
and the `cp_pad_len` trim in `prepare_model_outputs`.

In `utils.py`: the `torchtitan.components.dataloader -> components.data.loader`
import rename, `_import_torchtitan_model_module` (the experiments/ fallback),
`import os` + `VERL_TORCHTITAN_FLAVOR`, and the call-site swap to
`_import_torchtitan_model_module`.

Config: the `sequence_parallel`, `initial_load_path`, `pipeline_token_budget`
docstring lines, dataclass fields and the three yamls (the field only;
`pipeline_token_budget` is not read until patch 02).

Tests: `test_torchtitan_engine_receiver_names.py`;
`test_torchtitan_engine_cp_config.py` in a reduced form carrying
`TestParallelismCompatKwargs`, `TestEngineConfigDefaults` (keys
`sequence_parallel` / `initial_load_path` / `pipeline_token_budget`), `_BareModel`
and `TestFoldedTokenStreamProbe` (llama3 only), with no torchtitan-CP and no
kimi_k3 import.

## 02_engine_pipeline.patch  (+295 / -4, 9 hunks, 2 files)

    tests/workers/test_torchtitan_engine_pp_sync.py      +60/-0    1 hunk (new)
    verl/workers/engine/torchtitan/transformer_impl.py +235/-4    8 hunks

`_PipelineLossBridge`; `num_pp_microbatches=max(1, pipeline_parallel_size)` in
`ParallelismConfig`; the bridge install in `init_model` (`pp_schedule._loss_fn`);
the `pp_enabled` dispatch in `forward_backward_batch`; `_pp_forward_backward_batch`;
the `model_forward_step` "non-pipeline path" `RuntimeError` replacing the old
`NotImplementedError`; `_iter_pp_gathered`; `_FSDP_GRAD_UPCAST_GUARDED` +
`_guard_fsdp_grad_upcast`; the PP token-budget padding block of
`prepare_model_inputs` (`VERL_PP_TOKEN_BUDGET` / `pipeline_token_budget`,
`output_args["pp_pad_len"]`).

Test: `test_torchtitan_engine_pp_sync.py`.

## 03_engine_ep_lora_qat_sync.patch  (+923 / -34, 13 hunks, 4 files)

    tests/workers/test_torchtitan_engine_expert_stacks.py  +90/-0    1 hunk (new)
    tests/workers/test_torchtitan_engine_lora_sync.py     +114/-0    1 hunk (new)
    tests/workers/test_torchtitan_engine_peft_config.py   +283/-0    1 hunk (new)
    verl/workers/engine/torchtitan/transformer_impl.py    +436/-34  10 hunks

`import itertools`; the `torchtitan.config.transform` import (`apply_transforms`);
`_lora_transform`; the `iter_per_tensor_params_ep` import;
`training_kwargs["disable_cuda_graphs"] = True` (the EP token-dispatcher note);
`transforms = []` / the LoRA transform append / `apply_transforms`; the
`_lora_mode` guard and `_merged_state_dict_if_lora` in
`get_per_tensor_param_shard`; `_lora_mode`; the whole rewritten
`get_per_tensor_param` (`base_sync_done`, adapter vs merged state dict,
`sd_adapter` + the popped `expert_stacks`, `_wrapped_hf_base_names`, the adapter /
base halves, the EP gather through `iter_per_tensor_params_ep`, the generator
chain); `_iter_expert_stacks` (with the MX QAT fake-quant) and `_owning_module`;
`_titan_lora_wrappers`, `_is_lora_wrapper`, `_lora_prefix`, `_lora_factor`,
`_peft_config_from_wrappers`, `_wrapped_hf_base_names`, `_adapter_state_dict`,
`_warn_wrapped_bases_missing`, `_merged_state_dict_if_lora`.

Tests: `expert_stacks`, `lora_sync`, `peft_config`.

## 04_engine_context_parallel.patch  (+195 / -28, 19 hunks, 6 files)

    tests/workers/test_torchtitan_engine_cp_config.py           +110/-16  4 hunks
    verl/trainer/config/_generated_ppo_torchtitan_trainer.yaml    +3/-0    3 hunks
    verl/trainer/config/engine/torchtitan.yaml                    +3/-0    1 hunk
    verl/trainer/config/ref/torchtitan_ref.yaml                   +1/-0    1 hunk
    verl/workers/config/engine.py                                 +6/-0    3 hunks
    verl/workers/engine/torchtitan/transformer_impl.py           +72/-12   7 hunks

Drops `from torchtitan.distributed.context_parallel import
prepare_context_parallel_input` and widens the transform import to
`apply_transforms, ContextParallelTransform`; `_context_parallel_transform`
(flex Ulysses / all-gather-KV table, the vision-tower exclusion) minus its KDA
entry; `context_parallel_degree=1` in `ParallelismConfig` (replacing the direct
degree) with the "degree then transform" comment; the degree-raise +
`_context_parallel_transform` append in `__init__`; the CP gather in
`_finish_pred`; `pad_multiple = cp * 128` replacing the
`prepare_context_parallel_input` call; the `preprocess_inputs` block of
`prepare_model_inputs` and the `if -> elif self._folded_token_stream`; the
`context_parallel_backend` docstring line, dataclass field, `__post_init__`
assert and three yaml entries.

Tests: `test_torchtitan_engine_cp_config.py` grows to `TestContextParallelTransform`
and `TestDegreeThenTransformOrdering` (and `KEYS` gains
`context_parallel_backend`). **`TestMultimodalKwargs` is in patch 05, not here** --
it tests `_model_multimodal_kwargs`, which patch 05 introduces.

## 05_kimi_k3.patch  (+288 / -20, 27 hunks, 8 files)

    tests/workers/test_torchtitan_engine_cp_config.py      +26/-0    1 hunk
    tests/workers/test_torchtitan_engine_k3_processor.py   +68/-0    1 hunk (new)
    verl/utils/dataset/multiturn_sft_dataset.py            +24/-1    2 hunks
    verl/utils/dataset/rl_dataset.py                       +31/-1    2 hunks
    verl/utils/tokenizer/continuous_token_wiring.py         +3/-0    1 hunk
    verl/utils/tokenizer/tokenizer.py                      +59/-0    3 hunks
    verl/workers/engine/torchtitan/transformer_impl.py     +61/-14  14 hunks
    verl/workers/engine/torchtitan/utils.py                +16/-4    3 hunks

`utils.py`: the `kimi_linear` / `kimi_k3` model-type entries, the `dataset: Any =
None` field of `NoOpDataLoader.Config`, and the `text_config` nesting of the
shape lookup. `tokenizer.py`: the `KimiK3Processor` case, `_processor_takes_medias`
and `_call_medias_processor` (the pad-per-patch expansion) and the dispatch in
`build_multimodal_processor_inputs`. `multiturn_sft_dataset.py`:
`_processor_patch_size`. `rl_dataset.py`: `_images_from_messages` and the
`qwen_vl_utils` ImportError fallback. `continuous_token_wiring.py`: the `kimi_k3`
family entry. `transformer_impl.py`: `_MULTIMODAL_KEY_ALIASES`,
`_model_multimodal_kwargs`, the `model_mm_kwargs` call in `prepare_model_inputs`,
`batch.update(model_mm_kwargs)` inside the CP preprocessing,
`extra_inputs.update(model_mm_kwargs)` replacing the old
`extra_inputs.update(multi_modal_inputs)` TODO, the kimi_k3 KDA entry of
`_context_parallel_transform`'s backend map, and the Kimi-named comment /
docstring lines restored to their branch wording.

Tests: `test_torchtitan_engine_k3_processor.py`, plus `TestMultimodalKwargs`
appended to `test_torchtitan_engine_cp_config.py`.

## 06_metrics_logprob_diff.patch  (+6 / -0, 1 hunk, 1 file)

    verl/utils/debug/metrics.py  +6/-0  1 hunk

`rollout_logprobs_diff` and the two `training/rollout_logprobs_diff_{max,mean}`
metrics, with the 3-line comment that says why the probability-scale diff cannot
show the drift.

## local_env_and_diagnostics.patch  (+224 / -6, 18 hunks, 8 files) -- NOT for upstream

    tests/special_e2e/sft/run_sft_engine.sh                 +8/-2   3 hunks
    verl/third_party/vllm/__init__.py                       +8/-0   2 hunks
    verl/trainer/ppo/v1/trainer_base.py                     +6/-0   1 hunk
    verl/utils/debug/metrics.py                             +8/-0   1 hunk
    verl/workers/config/engine.py                           +1/-1   1 hunk
    verl/workers/engine/torchtitan/transformer_impl.py    +176/-2   8 hunks
    verl/workers/rollout/vllm_rollout/utils.py             +12/-0   1 hunk
    verl/workers/rollout/vllm_rollout/vllm_async_server.py  +5/-1   1 hunk

`_torch_accepts_bfx9` + `_fp32_matmul_emulation_optional` and the `with` around
`Trainer(self.config)`; `VERL_VLLM_VERSION` in `third_party/vllm/__init__.py` and
`vllm_async_server.py`; `partial_dtensor` in the `spmd_backend` assert; the
`run_sft_engine.sh` flags (`model.trust_remote_code`, `engine.spmd_backend=
partial_dtensor`, `engine.expert_parallel_size`); `KIMI_GRPO_DUMP_LOGPROBS`
(`metrics.py`, `trainer_base.py`), `KIMI_GRPO_DUMP_VLLM`
(`vllm_rollout/utils.py`), `KIMI_GRPO_FREEZE_SYNC`, `KIMI_GRPO_SYNC_CHECKSUM` and
`KIMI_GRPO_DUMP_SYNC` with their stderr prints; `_DYNAMO_PROBED` /
`_dynamo_probe_once` and its `VERL_TORCHTITAN_DYNAMO_PROBE`-gated call site.

Three items were put here that the task did not enumerate, because they are pure
diagnostics with no behaviour change and belong to no upstream capability:

- the `_plain_params_reported` offload warning (`transformer_impl.py`, 12 lines):
  names the non-DTensor parameters FSDP2 will refuse to move.
- `import sys` and `import contextlib` (only used by the above).
- the `if adapter_mode and sd_adapter is None:` -> `elif adapter_mode:` flip, see
  "could not separate cleanly" below.

Two items named in the task do not exist in this branch's diff and are therefore
in no patch:

- the `FlexibleArgumentParser` import fallback: `git diff upstream/main HEAD` does
  not mention `FlexibleArgumentParser` at all (both sides carry the same plain
  `from vllm.utils.argparse_utils import FlexibleArgumentParser`), so it is either
  already upstream or was dropped in the rebase.
- `enable_fsdp_gradient_division` ("the change if any"): there is none. The name
  appears in the diff only as a context line of the `utils` import list.

---

## What could not be separated cleanly (named hunks)

1. **`get_per_tensor_param`, the PP gather call site.**
   `_iter_pp_gathered` is patch 02, but the variable it wraps
   (`per_tensor_param`) only exists after patch 03 rewrites the function. Patch 02
   therefore wraps upstream's generator instead, in hunk
   `@@ -782,7 +910,74 @@` of `transformer_impl.py`:
   `return _gen(), None` becomes `gen = _gen()` / `if pp_enabled: gen =
   self._iter_pp_gathered(gen, device)` / `return gen, None`.
   Patch 03's hunk `@@ -867,53 +915,149 @@` then removes those four lines and
   re-adds the same call in the new structure (`if self.parallel_dims.pp_enabled:
   per_tensor_param = self._iter_pp_gathered(per_tensor_param, device)`), so five
   PP-shaped lines appear inside patch 03. The mechanism itself is entirely in 02.

2. **`ulysses_pad` is shared between TP and CP.**
   Patch 01 owns `pad_multiple`, the `ulysses_pad` call and the whole
   `if cp_pad_len:` block (position renumbering, label padding, mask rebuild),
   because the packed stream must divide by the TP degree with no CP at all.
   Patch 04 only adds `pad_multiple = self.parallel_dims.cp * 128`. The variable
   keeps its branch name `cp_pad_len` even in patch 01, where it is the TP pad.
   The base call's closing paren (a context line in the raw diff) is moved into
   patch 04's deletion so both stages stay syntactically valid.

3. **`if` vs `elif self._folded_token_stream:`** (`prepare_model_inputs`). Patch 01
   ships it as `if`; patch 04 changes the same line to `elif` when it adds the CP
   `preprocess_inputs` branch in front of it. One line, in 04's hunk
   `@@ -1601,7 +1639,29 @@`.

4. **`from torchtitan.config.transform import ...`.** Patch 03 needs
   `apply_transforms` (LoRA), patch 04 needs `ContextParallelTransform`. Patch 03
   adds the one-name import, patch 04 rewrites the same line to the two-name form
   (04's hunk `@@ -36,8 +36,7 @@`).

5. **`elif adapter_mode:` in `get_per_tensor_param`.** In HEAD this `elif` binds to
   the `if dump_dir ...` diagnostic block, so removing the diagnostics leaves a
   dangling `elif`. Patch 03 ships the standalone, correct form
   `if adapter_mode and sd_adapter is None:`; patch 0 flips it back to
   `elif adapter_mode:` when it re-inserts the diagnostic block. If the branch ever
   drops the diagnostics, patch 03's form is the one to keep (HEAD's `elif` makes
   the "no state-dict adapter" raise unreachable whenever `dump_dir` is unset).

6. **The `transforms` block ordering.** In the raw diff `self.trainer =
   Trainer(self.config)` is deleted (patch 0, for the BFX9 wrapper) before the
   `transforms` lines are added (patches 03/04). The block is moved ahead of the
   `Trainer(...)` line so that patch 03 alone applies the transforms before the
   Trainer is built.

7. **`_MULTIMODAL_KEYS` is in patch 01, not 05.** The task put it in 05, but
   `_squeeze_folded` (patch 01) reads it; leaving it in 05 makes patch 01 raise
   `NameError` on every folded-stream forward. `_MULTIMODAL_KEY_ALIASES` and
   `_model_multimodal_kwargs` stay in 05.

8. **`training_kwargs["disable_cuda_graphs"] = True`** is in patch 03 (it is the
   expert-parallel token-dispatcher constraint), although it is set
   unconditionally and is not part of the sync.

9. **Checkpoint `interval=1`** (the `save_freq` cadence fix) and the
   `components.dataloader -> components.data.loader` / `components.checkpoint ->
   components.checkpointer` / `components.lr_scheduler -> components.optimizer`
   import moves are in patch 01. They are tree-compat / generic-engine fixes the
   task did not assign; patch 01 is the first patch that has to build a Trainer.

10. **`text_config` in `utils.py`'s shape lookup** is in patch 05 (its own comment
    names kimi_k3 as the multimodal config it exists for), although the code is
    model-agnostic. Move it to 01 if the K3 PR should be map-only.

11. **Kimi-named comments.** Patches 01/03/04 carry generically worded versions of
    the comment lines that name Kimi K3, KDA or `kimi_k3_debugmodel_gated_lora`,
    and patch 05 rewrites those lines back to the branch's wording (9 one-to-few
    line hunks in `transformer_impl.py`). This is what makes the union bit-exact
    against HEAD; before filing, patch 05 should simply drop those restore hunks
    and the `_merged_state_dict_if_lora` docstring's measured-numbers paragraph
    (`Measured on kimi_k3_debugmodel_gated_lora: ... 151 both`), which the
    diff-audit rule keeps out of source entirely.

12. **`tests/workers/test_torchtitan_engine_cp_config.py`** is one file split over
    three patches, so patches 04 and 05 rewrite parts of what 01 created
    (`_helpers`, the module docstring, the imports, `KEYS`, the folded-stream
    subTest loop). 04's diff against 01's version is +110/-16.

13. **Not verified by execution.** No test was run and no GPU job was launched, as
    instructed. The only checks are structural: `ast.parse` and ruff F821/F811/F822
    on every intermediate tree, `git apply --check` per patch, and the empty final
    diff.

---

## Verification (transcript in `work/verify.txt`)

Scratch index only (`GIT_INDEX_FILE=.../verl_split/idx`); the repo's real index and
worktree were never touched (`git status` in `/tmp/wt_verl_0915` after the run
shows only the pre-existing untracked `verl_grpo_example_gsm8k_0217/`).

    $ cd /tmp/wt_verl_0915
    $ export GIT_INDEX_FILE=.../verl_split/idx && rm -f "$GIT_INDEX_FILE"
    $ git read-tree upstream/main
    $ for p in 01 02 03 04 05 06 local; do git apply --cached --check -p1 $p.patch; git apply --cached -p1 $p.patch; done
    apply --check OK: 01_engine_tp_packed_and_compat.patch
    apply --check OK: 02_engine_pipeline.patch
    apply --check OK: 03_engine_ep_lora_qat_sync.patch
    apply --check OK: 04_engine_context_parallel.patch
    apply --check OK: 05_kimi_k3.patch
    apply --check OK: 06_metrics_logprob_diff.patch
    apply --check OK: local_env_and_diagnostics.patch

    $ git diff --cached HEAD --stat
    (no output; `git diff --cached HEAD --stat | wc -l` = 0)

The intermediate-tree builder independently reports
`final tree 5787342848265d1ffcbd709eaa96d1231940009b == HEAD tree 5787342848265d1ffcbd709eaa96d1231940009b`.

String check, `grep -c -E "KIMI_GRPO_DUMP|VERL_VLLM_VERSION|_torch_accepts_bfx9|DYNAMO_PROBE|partial_dtensor"`:

    01_engine_tp_packed_and_compat.patch:0
    02_engine_pipeline.patch:0
    03_engine_ep_lora_qat_sync.patch:0
    04_engine_context_parallel.patch:0
    05_kimi_k3.patch:0
    06_metrics_logprob_diff.patch:0
    local_env_and_diagnostics.patch:26

The wider grep `KIMI_GRPO|SYNC-CHECKSUM|bfx9|BFX9` over patches 01-06 also returns
nothing.

Python validity of every intermediate tree:

    ast.parse: checked 23 files x 7 stages; failures: 0
    ruff --select F821,F811,F822, per stage 01..06 and 00: All checks passed!

## Files

    01_engine_tp_packed_and_compat.patch
    02_engine_pipeline.patch
    03_engine_ep_lora_qat_sync.patch
    04_engine_context_parallel.patch
    05_kimi_k3.patch
    06_metrics_logprob_diff.patch
    local_env_and_diagnostics.patch
    work/splitlib.py, work/assign.py, work/build.py     the generator (rerun: python3 work/build.py)
    work/cp_config_01.py, work/cp_config_04.py          staged versions of the cp_config test
    work/impl.items.txt                                 the line-item listing the assignment is keyed on
    work/verify.txt                                     the verification transcript

---
## Regenerated 2026-09-17 05:05 on head `549c1e21`

Head moved from `31079117` to `549c1e21` (four commits: the adapter-only refusal reattached to its own `if`, the gloo context-parallel tests, the Kimi K3 image path, the dataset fallback without qwen_vl_utils). `work/assign.py` was remapped by matching the engine file's line items between the two heads (2071 of 2081 matched; the moved `elif adapter_mode` block and the deleted measured-numbers docstring paragraph were the only differences):

- the `elif adapter_mode` refusal now sits inside patch 03 with no staged variant, since it attaches to 03's own `if`;
- the kimi-named measured-numbers paragraph is gone from 05 (deleted on the branch);
- new whole-file assignments: `tests/workers/test_torchtitan_engine_cp_gloo.py` (04); `verl/experimental/agent_loop/agent_loop.py`, `verl/workers/rollout/utils.py`, `verl/utils/tokenizer/__init__.py` (05);
- `verl/workers/rollout/vllm_rollout/vllm_async_server.py` splits into the version override (00, lines 74-79 of its items) and the media-block collapse (05, items 54 and 639).

The first cut's patches, manifest, assignment and item listing are kept in `work/prev_31079117/`.

    01_engine_tp_packed_and_compat.patch   8 files   +461 / -28
    02_engine_pipeline.patch               2 files   +295 / -4
    03_engine_ep_lora_qat_sync.patch       4 files   +923 / -34
    04_engine_context_parallel.patch       7 files   +516 / -28
    05_kimi_k3.patch                      12 files   +426 / -26
    06_metrics_logprob_diff.patch          1 file    +6 / -0
    local_env_and_diagnostics.patch        8 files   +223 / -5

`work/verify.txt` regenerated: every patch passes `git apply --check` on the previous stage, `ast.parse` and `ruff --select F821,F811,F822` are clean on every stage (27 files at the last one), the union reproduces HEAD exactly, and none of 01-06 carries a local marker (`KIMI_GRPO_DUMP`, `VERL_VLLM_VERSION`, `_torch_accepts_bfx9`, `DYNAMO_PROBE`, `partial_dtensor`) or a logbook path. Patch 05 now carries the whole image path (generic VL render, `media_features`, `collapse_media_blocks`, the banned pad) and its tests; patch 04 carries the gloo tests.

---
## Regenerated 2026-09-17 evening on head `2bf5fbc3` (base `1a8a0f5f`)

The branch was rebased onto upstream verl main `1a8a0f5f`; per file its diff against the base is byte-identical to before, so `work/assign.py` needed no remapping. `work/splitlib.py` now pins `BASE = "1a8a0f5f"` and `HEAD = "2bf5fbc3"` instead of reading `upstream/main` and `HEAD`, so the kit describes one pair of commits rather than whatever the worktree points at. Branch against base: 27 files changed, 2817 insertions(+), 92 deletions(-). The union of the seven patches reproduces the head exactly, every stage parses and is ruff-clean, and none of 01 to 06 carries a local marker (`work/verify.txt`).

---
## Regenerated 2026-09-17 evening on head `e8e3ba50`

Two commits were added to the branch after the previous regeneration: the pipeline token budget factored into `pipeline_token_budget()` with its tests, and the initial checkpoint source factored into `initial_checkpoint_source()` with its tests and the config-level check of `context_parallel_backend`. Inserting two functions near the top of the engine file shifted every later line item, so `work/assign.py` was remapped by matching items between `2bf5fbc3` and `e8e3ba50` (2055 of 2094 matched) and the 39 genuinely new items were assigned by hand: the checkpoint source and its call site to patch 01, the token budget and its call site to patch 02, and one delete-insert pair that is only diff alignment noise follows its neighbour. The two new test files go whole to 01 and 02. Branch against base: 29 files changed, 2943 insertions(+), 92 deletions(-).

`work/verify.txt` regenerated: each patch applies on the previous stage, `ast.parse` and `ruff --select F821,F811,F822` are clean at every stage, the union reproduces the head exactly, and none of 01 to 06 carries a local marker or a logbook path.

---
## Regenerated on head `7db90d7a`

The branch gained one more commit, the dynamo diagnostic that compiles `create_block_mask` in place (the probe that separated the environment from torch's cached wrapper while chasing the `allgather_kv` failure). It is diagnostic code, so all 31 of its new line items go to `local_env_and_diagnostics.patch`; two further new items at 785 and 786 are diff alignment noise over existing code and follow their neighbour into patch 01 rather than the diagnostic group. Branch against base: 29 files changed, 2974 insertions(+), 92 deletions(-).

One trap worth recording, since it cost a rebuild: the remapping must start from the assignment as committed for the previous head. Running it twice over an already-remapped table shifts every index a second time and the builder then reports several dozen unassigned items. `git checkout` the table first and check that it succeeded.

`work/verify.txt` regenerated: each patch applies on the previous stage, `ast.parse` and `ruff --select F821,F811,F822` are clean at every stage, the union reproduces `7db90d7a` exactly, and none of 01 to 06 carries a local marker or a logbook path.
