# Multimodal GRPO for Kimi K3 through the veRL torchtitan engine: scope and plan (2026-09-17)

Survey by a read-only subagent on 2026-09-17 (veRL fork `kimi_k3_integration_rebased`, torchtitan tree `k3_on_4025`, the source-built vLLM in `venv_verl`), checked against the files named. Step 10 of the remaining list.

## What exists

- vLLM: the source-built branch (`/workspace/vllm_k3` @ `26d687b321`, the same files in `venv_verl`) serves K3 with images end to end: `KimiK3ForConditionalGeneration` registered as multimodal (`vllm/models/kimi_k3/nvidia/model.py:1436-1450`), `KimiK3MultiModalProcessor` builds its own `KimiK3Processor` from the checkpoint's `AutoImageProcessor` (`mm_preprocess.py:99-143`), expands the placeholder to the per-image token count in `_get_prompt_updates` (`:295-330`), slices `pixel_values` per image by `grid_thws.prod(-1)`. The `-rel` export carries the processor files and `config.json:vision_config`; verl's rollout server already threads `multi_modal_data["image"]` into the sampling requests (`vllm_async_server.py:638-646`) and `rollout.limit_images` into `limit_mm_per_prompt`.
- Weight sync: the state-dict adapter maps every tower tensor (`state_dict_adapter.py:96-106`), `to_hf` re-fuses the tower's q / k / v into `wqkv` and reshapes the patch embedding; the engine's full-tensor sync runs `to_hf` on the whole dict, so the names line up with vLLM's mapper. The sharded delta export cannot carry the tower (a 3-to-1 fusion over local shards); multimodal runs use the full-tensor sync. The `rl` flavor already builds the tower, so today's text-only cells sync an untrained tower every step through the zero placeholder path.
- Engine (after `9b6775a2`): the processor's multimodal outputs are renamed to the forward's parameters (`grid_thws` to `grid_thw`), filtered by the forward signature, `special_tokens={"image_id": media_placeholder_token_id}` added from the HF config when images are present, multimodal tensors keep their batch axis through the folded-stream squeeze (a one-image micro-batch's `grid_thw` is `[1, 3]`), and under CP they go through `preprocess_inputs` with the stream, which builds `vision_bank_indices_T` and shards it. Before that commit the dict was splatted raw into the forward.

## What is missing (in order)

1. The training-side token stream. `KimiK3VisionProcessor.make_image_prompt` emits one `<|media_pad|>` per image; the expansion to `(H // kh) * (W // kw)` pads is vLLM's (`mm_preprocess.py:295-330`, `media_tokens_calculator`). verl's `build_multimodal_processor_inputs` (`verl/utils/tokenizer/tokenizer.py:102-134`) calls `processor(text=..., images=...)`, which `KimiK3Processor.__call__` answers by tokenising the text alone when `messages` and `medias` are both None (`kimi_k3_processor.py:128-131`): the images are dropped silently, no `pixel_values`. The call must be `processor(text=..., medias=[{"type": "image", "image": img}, ...])`, and verl must expand the pads itself to the count `get_vision_positions` / `build_vision_bank_indices` require.
2. A continuous-token builder family for `kimi_k3` (`continuous_token_wiring.py:112-142` has `kimi_vl`, not `kimi_k3`; the default VL builder is the one making the wrong processor call). The rollout side needs the opposite convention: prompt ids with the un-expanded `<|kimi_image_placeholder|>` (one per image), since vLLM expands. Two spellings, both tested.
3. The dataset's vision loader: `RLHFDataset.process_vision_info` imports `qwen_vl_utils`, absent from `venv_verl`; a subclass with `process_multi_modal_info` (the hook the agent loop prefers, `agent_loop.py:300-307`) is the right answer, since `fetch_image`'s smart resize is Qwen's, not NaViT's.
4. Data: no image dataset on the box (`pixparse/cc12m-wds` in the HF cache is the README only). A small parquet of synthesised images (solid colours / digit tiles) with `prompt` carrying `<image>`, `images` as bytes, and a reward on an image property, on the shape of `examples/data_preprocess/geo3k.py` and `matrix_scripts/synthetic_reward.py`.
5. The run: `data.image_key=images`, `rollout.limit_images=1`, `max_prompt_length` above the expanded length, dp only for the first green run, the full-tensor sync.

## Risks named by the survey

- Patch ordering parity: torchtitan's `vision_to_patches` takes `patch_order` ("block" / "raster") and MoonViT3d wants "raster"; the released `navit_patchify` has no parity test (`test_kimi_k3_vision_preprocess_parity.py` covers resize and token counts only). A wrong order gives plausible numbers and a wrong policy; one image through both paths, embeddings compared, settles it before any cell is trusted.
- `spmd_types` on un-annotated vision tensors outside CP (the engine annotates only through `preprocess_inputs`, only under CP): whether `spmd.assert_type` at `model.py:1013-1023` accepts them under TP needs a run.
- vLLM's `PromptReplacement` on pre-tokenised ids, `qwen2_5_vl_dedup_image_tokens` on a K3 processor (`vllm_async_server.py:637`), and which `config.json` is live in the `-rel` export (three variants on disk; `media_placeholder_token_id` 163605 must equal `<|media_pad|>`).
- GPU budget: the text cells run two 16 GB cards at `gpu_memory_utilization` 0.35; the tower and image activations on top are untested.

## Addendum, 2026-09-17 evening: the image path read against every parallelism axis

Code read before the cells, so a failure can be told from a design gap. The engine has three routes for the vision tensors and each one was followed to the model:

- **No context parallel**: `_model_multimodal_kwargs` renames the processor's keys, keeps what the forward takes, adds `special_tokens={"image_id": ...}`, and the dict joins `extra_inputs` at the end of `prepare_model_inputs`. The model's `_prepare_multimodal_embeds` then runs the tower and scatters its features at the placeholder positions `get_vision_positions` finds.
- **Context parallel**: the same dict goes into the batch handed to the model's `preprocess_inputs`, which builds `vision_bank_indices_T` and shards it with the stream; the engine clears its own copy so the tensors are not passed twice. The model takes the bank-index path (`gather_vision_embeds`) instead of the scatter.
- **Pipeline**: `prepared.append((index, input_ids, {**extra_inputs, **extra_kwargs}))`, so every stage's forward receives the vision kwargs and only the stage that owns `tok_embeddings` acts on them; the others accept and ignore them.

Two interactions that looked like they could break the placeholder alignment, both read and found sound:

- The pipeline's token-budget padding appends `pad_token_id` tokens at the **end** of the packed stream and continues the positions; the media placeholders sit inside the stream and their count is unchanged, and the pad id (0) is not the media pad (163605), so `get_vision_positions` counts the same positions. The bridge cuts the logits back before the loss.
- The tensor-parallel padding (`pad_multiple = lcm(cp * 128, tp)`) pads the same way and `_finish_pred` gathers only the vocabulary or sequence shard of the logits; no vision tensor passes through either.

So no engine change is predicted for the image path under TP, PP or CP by itself. The open risk stays the one the survey named: `spmd.assert_type` on the vision tensors outside CP (the model declares the token layout replicated on TP for the scatter), which only a run can settle. The cells of `matrix_scripts/verl_img5d.sh` are what settles it.

## Image cells across the parallelism axes (2026-09-17 evening, 8 x RTX 5060 Ti)

`matrix_scripts/verl_img5d.sh`, two chains on disjoint GPU sets (two Ray clusters, the box's limit): the colour parquet, 8 prompts at n=2, 3 steps, the `rl` flavor on tree `/tmp/wt_int0916_rl`, engine `549c1e21`. Each cell is the image GRPO run under one parallelism shape; a cell counts as passing when its three steps complete (rc 0) and its rollout-vs-actor log-prob diff stays in the class the text cells read on this debug model (0.13 to 0.16).

    cell     shape            rc  steps  logprob diff mean
    img_tp2  tp 2             0   3      0.14548
    img_cp2  cp 2             0   3      0.13497

Both carried images rather than dropping them silently: `prompt_length/mean` is 111.0 in each, the same figure the dp2 image cell of the morning read and exactly what the builder produces offline for one row of this parquet (the expanded media block, four pads for a 56x56 image, plus the text); `data.image_key=images` and `return_multi_modal_inputs=True` are in both configuration dumps, and the grad norms (3.81 to 4.06) sit in the dp2 image cell's class (3.88 to 4.00). That is still indirect: the direct check is the drop-images diagnostic (`matrix_scripts/verl_drop_images.patch`, `KIMI_GRPO_DROP_IMAGES`), which runs the same batch text-only, and it waits for a free pair of GPUs.

Both pass, which is what the code read predicted: the tensor-parallel padding and the logit gather never see a vision tensor, and under context parallel the model's own `preprocess_inputs` shards the vision bank with the stream. The `spmd.assert_type` risk the survey named for the vision tensors outside context parallel does not fire under TP.

Reporting note: the runner's error filter greps the log for "Error", which also matches the trainer's configuration dump (`'truncation': 'error'`), so a passing row can carry that text; rc and the step count are the verdict. Fixed after the chains finish, since a running bash script must not be edited.
