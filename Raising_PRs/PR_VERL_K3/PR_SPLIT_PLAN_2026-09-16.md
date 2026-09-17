# veRL fork: what the K3 work changed, and how it splits into upstream PRs (2026-09-16)

Branch `kimi_k3_integration_rebased` = `a3661ae3`, 46 commits on upstream verl `00cd5b44` (2026-09-15). Upstream `verl-project/verl` main is `67858929` (2026-09-16), four commits ahead, two of them touching the engine file (#7873 gc during refit, #7874 detach model_output); the rebase is small. Upstream's torchtitan engine (`verl/workers/engine/torchtitan/transformer_impl.py`, 884 lines) has no LoRA, no EP, no CP, no Kimi, and refuses pipeline parallelism.

## Inventory: 16 files, +1819 / -65

| group | files | lines (approx.) | goes upstream |
| --- | --- | ---: | --- |
| K3-specific | `engine/torchtitan/utils.py` (model_type map `kimi_linear` / `kimi_k3` to the package, the `experiments/` import fallback, `VERL_TORCHTITAN_FLAVOR`, the no-op dataloader's `dataset` field); `transformer_impl.py` three flags keyed on `torchtitan_name == "kimi_k3"` (contiguous rank-ordered CP shards for the head-tail balancer, module-internal CP, folded `[T]` token stream), the packed document offsets and the model's own mask builder, the token-dispatcher note; `utils/tokenizer/tokenizer.py` (`KimiK3Processor` case, no rope index); `utils/dataset/multiturn_sft_dataset.py` (`patch_size` read tolerant of `media_proc_cfg`) | about 100 | yes, last, after the engine PRs; smaller still once the three flags become capability probes |
| generic engine capability | `transformer_impl.py`: pipeline parallelism (`_PipelineLossBridge`, `_pp_forward_backward_batch`, token-budget padding, one verl micro-batch per stage, every rank ships every stage's tensors in the sync, the FSDP2 grad-upcast guard), tensor parallel on the packed stream (padding to the TP degree, vocab-sharded logits gathered without gradient scaling), context parallel over the packed no-padding stream through verl's ulysses helpers, expert stacks gathered whole before the HF split, LoRA sync on torchtitan core's LoRA (`peft_config`, adapter-only sync, merge before sync, wrapper-segment naming; ten helpers), fake-quantized expert stacks for QAT models, `ParallelismConfig` compat kwargs, flavors defined as `config_registry` functions, dp-mesh lookup without an fsdp mesh, checkpoint interval from `save_freq` | about 1000 | yes, as model-agnostic PRs, each testable on upstream's own models |
| tests | `tests/workers/test_torchtitan_engine_{peft_config,lora_sync,expert_stacks,receiver_names,pp_sync}.py` | 610 | with their PRs |
| environment | BFX9 matmul-emulation opt-out (`_torch_accepts_bfx9`, `_fp32_matmul_emulation_optional`), `VERL_VLLM_VERSION` override (`third_party/vllm/__init__.py`, `vllm_async_server.py`), `FlexibleArgumentParser` import fallback, `partial_dtensor` in `config/engine.py`, the sft e2e script flags | about 40 | no; a local patch, or one separate small PR for the version override if wanted |
| diagnostics | `KIMI_GRPO_DUMP_LOGPROBS` / `KIMI_GRPO_DUMP_VLLM` hooks (`utils/debug/metrics.py`, `rollout/vllm_rollout/utils.py`, `trainer/ppo/v1/trainer_base.py`), the sync checksum and freeze notice on stderr | about 45 | no; strip before any PR. The log-probability rollout-vs-actor diff metric next to the probability one (`metrics.py`, 4 lines) is generic and can go on its own |

So the K3-specific surface in verl is about 100 lines, and it sits at the verl-to-torchtitan interface (model package lookup, flavor lookup, the three model traits, the processor). The other 1000 lines are engine capabilities the model needed but that name no model.

## Before a draft PR

1. Rebase onto `67858929` (two engine-file commits to merge over).
2. Strip the diagnostics group and park the environment group in a local patch (`matrix_scripts/` or a kit patch file), the way the KDA guard lift is parked on the torchtitan side.
3. Replace the three `torchtitan_name == "kimi_k3"` checks with capability probes on the torchtitan model or its spec (does the model take a folded `[T]` stream; does it run CP inside its modules; does its CP need contiguous rank-ordered shards). Then the engine PRs carry no model name and the K3 PR is the map, the processor case and the patch-size read.
4. The CP piece is written for the pre-4639 module-internal Ulysses (the `13a84361` / `27b72cc3` line) and refuses packed micro-batches under CP; on main tip K3's CP is 4639's (`preprocess_inputs`, `kda_cp_routing`). The CP PR either waits for that port or is written against main's CP API from the start; shipping the old path upstream and then replacing it is worse than shipping it later.

## Stack

1. engine: tensor parallel on the packed stream, and the flavor / `ParallelismConfig` / dp-mesh compat pieces (tests: receiver names).
2. engine: pipeline parallelism (`_PipelineLossBridge`, token budget, sync from every rank; test: pp_sync). Verifiable on upstream's llama / qwen debug models, which is what the reviewer will run.
3. engine: expert-stack gather in the sync, LoRA sync on torchtitan core's LoRA, fake-quantized expert stacks (tests: expert_stacks, lora_sync, peft_config).
4. engine: context parallel over the packed stream, once on main's CP API.
5. Kimi K3: the model-type map, `KimiK3Processor`, the patch-size read, the folded-stream and mask hooks as capability probes (with the trees named: torchtitan main after #4025 / #4499, the fork tree for PP / CP / LoRA).
6. optional: the log-probability diff metric.

Each engine PR's body states which torchtitan tree it was run against; the K3 PR names the open torchtitan PRs it needs for PP / CP / LoRA and stays a draft until they merge.

## Addendum, 2026-09-17

- The engine commits of the night (`7b78ad99` CP port, `bfb5a5e6` DCP initial load, `dd292978` / `2e3b127f` / `a2ef3b93` the sequence_parallel, context_parallel_backend and initial_load_path fields, `580b2ef0` adapter-only naming through `to_hf`) carry no trailers; 32 of the 46 older commits of the branch carry `Co-Authored-By` / `Claude-Session` lines from earlier sessions. The split rewrites every commit anyway (step 2 of "Before a draft PR"), which is where those lines go.
- The engine cells that pass on the ported engine: cp2, fsdp2 x pp2, tp2 x ep2, cp2 x tp2 (all 3 steps, rc 0, fsdp2 grad-norm class); the QLoRA packed-base sync needs `initial_load_path` (a packed DCP from `scripts/quantize_lora_dcp.py`) and the merged sync, and the tree's merge had to learn a locally sharded packed base (`K3_INT_20260916.md`, the 09-17 section).
- Two findings that belong to the K3 PR, not the engine PRs: the image-free placeholder path (`add_zero_valued_dependency`, tree `43ad3bfc2`) and the fused `w13` LoRA target (`0be1fee6f`).

## What remains on the veRL side (2026-09-17)

Before any PR:
1. Strip and split (this plan's stack): drop the diagnostics, park the environment hacks, rewrite the history without trailers, one model-agnostic PR per capability.
2. Run each engine capability on an upstream model (llama3 / qwen3 debug models through the same e2e runner): PP, TP on the packed stream, CP, the LoRA sync. Nothing but Kimi K3 has been through this engine; a reviewer runs their own model first.
3. A CP unit test (cp=2, gloo, a packed two-document micro-batch through `prepare_model_inputs`, gathered logits equal to cp=1) and a same-batch gradient check of the CP gather's scaling (today's evidence is the grad-norm class only).
4. Config fields with tests: `context_parallel_backend`, `sequence_parallel`, `initial_load_path`; `VERL_PP_TOKEN_BUDGET` from the environment into the config.
5. Locate why the colocated worker cannot compile the all-gather-KV backend's BlockMask sharding (dynamo off in the worker), or document Ulysses as the supported backend.

Engine coverage still missing:
6. LoRA + CP (transform ordering: the flavor's `LoRAConverter` runs before the engine's `ContextParallelTransform`); untested, likely broken.
7. PP + CP and PP + EP together (the bridge's CP gather, the sync from every stage with expert stacks); three-axis cells (fsdp2 x tp2 x ep2, cp2 x pp2).
8. QLoRA: the fused `w13` projection under the packed layout (split `w1` / `w3` serialization against `w13.*` packed keys); the packed DCP has to be written by the engine venv's torch; the adapter-only path with a fused projection.
9. Save and resume through the engine (verl `save_freq`, titan interval 1): a resume cell.
10. Multimodal GRPO: the engine's "multimodal not yet supported" path (pixel values through `prepare_model_inputs`, the K3 processor, a multimodal reward and data); the image-free placeholder path is fixed for TP now.

Numbers and venue:
11. A real-weight cell (the released checkpoint or the 48B graft): the engine's rollout-vs-actor log-prob diff of 0.15 on the debug model means nothing; Miles quotes a 2e-3 KL floor at 2.8T.
12. Throughput and memory per cell for the bodies (today's cells are 3-step smokes).
13. The vLLM side: the engine runs on a source-built vLLM branch behind `VERL_VLLM_VERSION`; pin it against vLLM's own K3 support and document the export (`-rel`).
14. The post-training venue decision (torchtitan RL #4680 or the veRL engine), the RFC's open question 2, before the K3 PR is filed on verl.

