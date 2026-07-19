# veRL torchtitan backend — verified against source (2026-07-19)

> Resolves PLAN §0a #5 / HANDOFF's "veRL native = FSDP/Megatron only 🔧"
> conflict. Checked out `volcengine/verl@e003163` (2026-07 HEAD, shallow)
> on the 8x5090 box and read the engine source directly.

## Verdict

**The handoff is outdated: veRL ships a first-class torchtitan model
engine.** `strategy: torchtitan` sits alongside dp/megatron/mindspeed/
veomni under `verl/trainer/config/model_engine/`. The conservative
"major integration work" planning case is wrong in our favor — the work
is model *registration*, not engine construction.

Source anchors (all under `verl/workers/engine/torchtitan/`):

- `transformer_impl.py` (766 lines): builds a real titan
  `Trainer.Config` — `OptimizersContainer.Config` w/ param groups,
  `LRSchedulersContainer.Config`, `ParallelismConfig`
  (dp_replicate/dp_shard/tp/**pp**/cp/ep + spmd_backend),
  `CheckpointManager.Config`, Selective/Full AC, CompileConfig.
  This tracks the *current* Configurable-based titan API — the same
  API our fork just merged. Version pairing is favorable.
- `utils.py`: `derive_torchtitan_name_and_flavor(hf_config)` maps
  `hf_config.model_type` via `_HF_MODEL_TYPE_TO_TORCHTITAN_NAME`
  (qwen2/qwen3/qwen2_moe/qwen3_moe/llama/llama4/deepseek_v3/gpt_oss),
  imports `torchtitan.models.<name>`, calls
  `model_registry(flavor, attn_backend=...)`, and matches the flavor by
  (dim, n_layers, vocab_size) against any module-level `*_configs` dict.

## What this means for kimi_k3 (gap list, in work-item order)

1. **HF model_type mapping**: `kimi_linear` is not in
   `_HF_MODEL_TYPE_TO_TORCHTITAN_NAME` -> small veRL-side patch (one
   dict entry + upstreamable PR later).
2. **Import namespace**: the engine imports `torchtitan.models.<name>`,
   our folder is `torchtitan.experiments.kimi_k3`. Either extend the
   veRL patch to try `torchtitan.experiments.` as fallback, or wait for
   the models/ promotion. Recipe-side patch is the pre-7.27 answer.
3. **model_registry signature**: engine passes
   `attn_backend=engine_config.attn_type`; our
   `kimi_k3.model_registry(flavor)` must tolerate an `attn_backend`
   kwarg (accept-and-ignore or dispatch).
4. **Flavor discovery**: engine looks for a module-level `*_configs`
   dict; kimi_k3 exposes flavor functions instead -> add a
   `kimi_linear_configs`-style dict (or extend the discovery in the
   veRL patch).
5. **HF weight load**: engine sets `initial_load_in_hf=True,
   initial_load_path=<HF path>` -> goes through titan checkpointing's
   state_dict_adapter. **The phase11 hf_to_dcp promotion (work item 1)
   is on the veRL critical path too**, not just continued-pretrain.

## PP status inside the engine (matches PLAN §2 posture)

- Training path (`forward_backward_step`-equivalent): has
  `pp_enabled` branches wired via titan `parallel_dims`.
- **`model_forward_step` raises NotImplementedError under PP** ("will
  be implemented in a follow-up PR") -> recompute-logprob flows (GRPO
  ref/actor logprob) cannot use PP today. Plan stays: 48B post-train on
  FSDP+TP+EP without PP; PP-post-training gap only bites at K3 scale.
- FSDP offload helpers (`load_fsdp_model_to_gpu` /
  `offload_fsdp_model_to_cpu`) are wired for param offload between
  rollout and train phases; checkpoint save overrides titan's folder.

## Next actions (feeds PLAN item 3 / task list)

- [x] Minimal veRL recipe patch -- DONE same day:
      [`verl_kimi_linear_engine.patch`](verl_kimi_linear_engine.patch)
      (2 hunks: kimi_linear->kimi_k3 mapping + experiments-namespace
      fallback) + fork-side shims (attn_backend kwarg, dim/vocab_size
      duck props, kimi_linear_configs flavor dict). Verified standalone:
      the patched matcher resolves the official 48B HF config to
      ("kimi_k3", "kimi_linear_48b_baseline") through the real registry.
- [x] state_dict_adapter promotion -- DONE (603/603 keys, bit-exact
      sharded load of the official weights; see GPU_CHECKLIST_RESULTS).
- [ ] LoRA P0 trio (veRL PEFT config engine-agnostic; verify the
      torchtitan engine honors it) + full veRL install + 48B LoRA SFT
      smoke -- next leg.
