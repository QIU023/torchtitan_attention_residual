# Replacing the stubbed GRPO sampler: where it stands

The GRPO loop works end to end with a stubbed sampler -- deterministic token ids
rather than model samples -- on single GPU, dp_shard=2 and cp=2, through both the
direct module path and veRL's `forward_backward_batch`. What remains is
generation.

## veRL's torchtitan engine has no generation path

`TorchTitanEngineWithLMHead` exposes `train_batch`, `infer_batch`,
`forward_backward_batch`, `forward_step` and the optimizer/checkpoint methods.
There is no `generate` and no rollout entry, so sampling has to come from a
serving stack -- for this project, vLLM.

## vLLM can serve the architecture; our mini config is inconsistent

The `kimi-k3` branch registers `KimiK3ForConditionalGeneration`, exactly what
`/workspace/k3mini_hf/config.json` declares, and the checkpoint has real weights
(946 tensors, 162 MB, non-zero). Loading still fails:

    AttributeError: property 'hidden_size' of 'KimiK3Config' object has no setter

The cause is the config we generated, not vLLM. `config.json` sets
`model_type: kimi_k3` and `architectures: [KimiK3ForConditionalGeneration]` --
both multimodal -- while `auto_map.AutoConfig` points at
`configuration_kimi_k3.KimiLinearConfig`, the TEXT-ONLY config. transformers
resolves the text config and says so ("You are using a model of type `kimi_k3`
to instantiate a model of type `kimi_linear`"), so vLLM receives a flat config
where it expects one nested under `text_config`, and the `hidden_size` write on
the multimodal wrapper fails.

Fixing it means either pointing `auto_map` at the multimodal `KimiK3Config` with
the text fields nested under `text_config`, or changing `architectures` to the
text-only class. Both change what the checkpoint claims to be, so the
state_dict_adapter key mapping has to be re-checked against whichever is chosen.
Not a one-line edit.

## What is and is not blocked

Not blocked: the GRPO objective, reward extraction, group-relative advantages,
the optimizer step, and all three verified parallel configurations. None depend
on the sampler.

Blocked: sample quality, and anything needing real rollouts -- KL to a reference
policy, entropy, sequence-level rewards on generated text.
