# The vLLM the engine runs on, and what the `-rel` export is (2026-09-17)

Item 13 of the veRL remaining list ("pin it against vLLM's own K3 support and document the export"). Everything below is measured on this box, not read off a README.

## The build that actually runs

`import vllm` inside `venv_verl` resolves to `venv_verl/lib/python3.12/site-packages/vllm`, a plain non editable copy (no symlink, no `__editable__` pth for vllm), installed 2026-09-02. Its version string is `0.1.dev1+g6dc76a9ad`, so it was built from upstream `6dc76a9ade` (2026-07-28), the day after vLLM announced day-0 Kimi K3 support.

The important consequence for the PR story: **the K3 support the engine uses is upstream vLLM's own**, not a private implementation. That build carries 49 Kimi K3 python files under `vllm/models/kimi_k3/` with `nvidia/` and `amd/` variants, exposing `KimiK3ForConditionalGeneration`, `KimiK3MTP` and `KimiK3MultiModalProcessor`, and taking `pixel_values` plus `grid_thws` as its multimodal inputs. The same names and the same module path appear in vLLM's published docs, so the rollout side of every cell is upstream behaviour.

## What the checkout adds, and why it is not in the running build

`/workspace/vllm_k3` sits on branch `kimi_k3_supports_lora` at `26d687b321`, four commits ahead of the build's base, all from 2026-08-11 to 2026-08-14: `KimiLinearForCausalLM` declares LoRA support while excluding the KDA layers, the loader names its candidates on a `params_dict` miss and scopes the LoRA probe, the reason LoRA skips KDA is corrected, and the LoRA prefix is read from the layer rather than from parameter names. Comparing the two trees file by file, the running copy differs from the checkout in exactly the two files those commits touch (`models/kimi_k3/nvidia/model.py` and `model_executor/layers/fused_moe/routed_experts.py`), so the running build predates them.

That does not invalidate any cell that has run. Every LoRA and QLoRA cell used the merged weight sync (`model.lora.merge=True`), which folds adapters into the base before the sync, so vLLM never receives an adapter and the declaration those four commits add is never exercised. The build matters only for an adapter-only rollout, which is not what any recorded cell did. Anyone wanting that path has to reinstall from the checkout first.

## Why `VERL_VLLM_VERSION` is set at all

The fork gates on a minimum vLLM version (`verl/third_party/vllm/__init__.py:37` reads the override, `vllm_async_server.py:79` parses it). A source build reports `0.1.dev1+g6dc76a9ad`, which `version.parse` orders below the `0.18.0` the gate wants, so every runner exports `VERL_VLLM_VERSION=0.18.0`. It is a parse workaround for a dev version string, and it makes no claim about which features are present. Pinning properly means naming the upstream commit, which is what this file does.

## What the `-rel` export is

Two exports sit on disk and they are far closer than their names suggest. `config.json` is byte identical between them. Both hold 1428 tensors, the same key set (nothing is only in one) and the same dtypes.

Exactly nine tensors differ, and they differ only in shape: `language_model.model.layers.N.self_attn.A_log` is `[16]` in `/root/models/kimi-k3-debug-nt` and `[1, 1, 16, 1]` in `/root/models/kimi-k3-debug-nt-rel`. Nine of the twelve layers carry KDA, which is why there are nine. The rank-4 spelling is the one the released `modeling_kimi_linear.py` writes and the one vLLM's KDA weight loader expects.

So `-rel` means released layout, and it is the export every rollout bearing runner names: `verl_grpo_int0915.sh`, `verl_grpo_int0915_nd.sh`, `verl_grpo_int0916.sh`, `verl_grpo_int0916_nd.sh` and `verl_grpo_k3_image.sh` all default `MODEL_PATH` to it. Every number reported from those cells, the ten image cells included, was produced against `-rel`. The plain export is named by the older runners and by paths that never start a rollout engine.

The file sizes differ by 56 bytes and the checksums differ, which is the nine reshaped headers and nothing else.

## What to carry into a PR body

One sentence is enough and it is now defensible: the engine's rollout side runs upstream vLLM's Kimi K3 support (built from `6dc76a9ade`), the checkpoint it loads is the released layout export, and `VERL_VLLM_VERSION` is a version string workaround for a source build rather than a feature gate.
