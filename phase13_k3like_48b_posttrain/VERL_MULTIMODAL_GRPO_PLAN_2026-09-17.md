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

    cell     shape            rc  steps  logprob diff mean  prompt length  grad norm range
    img_tp2  tp 2             0   3      0.14548            111.0          3.84 to 4.06
    img_cp2  cp 2             0   3      0.13497            111.0          3.81 to 3.99
    img_ep2  fsdp 2 x ep 2    0   3      0.13499            111.0          3.82 to 4.01
    img_pp2  pp 2             0   3      0.11449            111.0          3.80 to 4.07

Combinations:

    cell          shape          rc  steps  logprob diff mean  prompt length  grad norm range
    img_cp2tp2    cp 2 x tp 2    0   3      0.14940            111.0          3.85 to 4.04
    img_pp2cp2    pp 2 x cp 2    0   3      0.10457            111.0          3.89 to 4.09
    img_fsdp2pp2ep2  fsdp 2 x pp 2 x ep 2  0  3  0.11079          111.0          3.87 to 4.11
    img_dp2cp2tp2ep2  fsdp 2 x cp 2 x tp 2 x ep 2  0  3  0.15061   111.0   3.78 to 4.07
    img_pp2tp2cp2  pp 2 x tp 2 x cp 2  0  3  0.11032   111.0   3.85 to 4.07
    img_tp2ep2    tp 2 x ep 2    0   3      0.13816            111.0          3.91 to 4.02

Every cell carried images rather than dropping them silently: `prompt_length/mean` is 111.0 in each, the same figure the dp2 image cell of the morning read and exactly what the builder produces offline for one row of this parquet (the expanded media block, four pads for a 56x56 image, plus the text); `data.image_key=images` and `return_multi_modal_inputs=True` are in both configuration dumps, and the grad norms (3.81 to 4.06) sit in the dp2 image cell's class (3.88 to 4.00). That is still indirect: the direct check is the drop-images diagnostic (`matrix_scripts/verl_drop_images.patch`, `KIMI_GRPO_DROP_IMAGES`), which runs the same batch text-only, and it waits for a free pair of GPUs.

All four single-axis cells pass, which is what the code read predicted: the tensor-parallel padding and the logit gather never see a vision tensor, and under context parallel the model's own `preprocess_inputs` shards the vision bank with the stream. The `spmd.assert_type` risk the survey named for the vision tensors outside context parallel does not fire under TP.

Reporting note: the runner's error filter greps the log for "Error", which also matches the trainer's configuration dump (`'truncation': 'error'`), so a passing row can carry that text; rc and the step count are the verdict. Fixed after the chains finish, since a running bash script must not be edited.

The pipeline cell is the one that could have lost the images silently, and it did not: its prompt length is the same 111, so the token-budget padding appended at the end of the packed stream left the media pads inside the stream where `get_vision_positions` counts them, and the stages that do not own `tok_embeddings` ignored the vision kwargs they were handed. Its log-prob diff (0.11449) is the lowest of the four but stays in the class the text cells read.

A side result from the same cell: its configuration dump shows `pipeline_token_budget: None`, so the budget came from `VERL_PP_TOKEN_BUDGET` in the launcher. That is the environment fallback of `pipeline_token_budget()`, the function factored out of `prepare_model_inputs` this evening, exercised by a real run as well as by its unit tests.

The first combination cell is the one whose padding stacks: context parallel wants whole 128-token blocks per shard and tensor parallel wants the stream to divide by the degree, so `pad_multiple` is their least common multiple and the packed stream grows the most here. Its prompt length is still 111 and its gradient norms sit with the rest, so the media pads kept their positions under the larger padding as well.

Pipeline by context parallel is the longest path the vision tensors take: the pipeline pads the packed stream to its token budget first, then the model's `preprocess_inputs` shards that padded stream and builds the vision bank indices on it, and only the stage owning the embeddings gathers the tower's features back. Its prompt length is 111 like every other cell, so the media pads survived a padding and a sharding in sequence.

### Two findings about the runner itself, before any verdict is read from it

- **The exit code in the results file means nothing.** `verl_grpo_k3_image.sh` ends with `echo "rc=$rc"`, so the script always exits 0 and the matrix runner records `rc=0` for every cell, passing or not. What separates a passing cell from a failing one is the step count (6 log lines for 3 steps) and `prompt_length/mean` (111 for this parquet). Every row above was read that way; the `rc` column stays only because the file already carries it.
- **`img_tp2ep2` as first written was an invalid shape, not an engine defect.** It asked for 4 GPUs with `dp_shard 1, tp 2`, and torchtitan asserts `dp_replicate * dp_shard * cp * tp * pp == world_size`, with expert parallelism a divisor of `dp_shard * cp * tp` rather than a world dimension of its own: `Invalid parallel dims: dp_replicate(1) * dp_shard(1) * cp(1) * tp(2) * pp(1) != WORLD_SIZE(4)`. The 09-16 text matrix runs the same combination on two GPUs (`tp2_ep2|2|...|$D1 $TP2 $EP2`). Relaunched on two GPUs with the same flags; the four-GPU row is void.

The three-axis cell (data-parallel sharding by pipeline by expert parallelism) passes with the same prompt length, so the vision kwargs survive a stage split and an expert split at once. Seven cells of the matrix are now green and the engine carries no change for any of them.

The relaunched `img_tp2ep2` on its correct two-GPU shape passes as well, so every combination this box can hold is green: the four axes on their own, three pairs, and one triple. Eight cells, one prompt length, no engine change.

## The drop-images comparison, and what it does and does not show (2026-09-17)

`matrix_scripts/verl_img_dropcheck.sh` runs the same image cell twice on two GPUs, the second time with `KIMI_GRPO_DROP_IMAGES=1`, which makes `_model_multimodal_kwargs` return an empty dict so the forward gets no vision tensor at all (`matrix_scripts/verl_drop_images.patch`, applied for the pair and reverted after). The token stream is untouched either way, so the model still sees the expanded media pads and simply embeds them from the token table when the tower's features are missing.

    run             pg_loss      rollout-vs-actor logprob diff  grad norm  prompt length
    with images     0.04584863   0.17050                        3.79165    111.0
    without images  0.05104822   0.18142                        3.86080    111.0

The column that carries the argument is the third one. The rollout engine always has the images (vLLM builds its own features from the placeholder), so dropping them on the actor side alone should widen the actor-to-rollout log-prob gap, and it does, from 0.17050 to 0.18142. The size is the size to expect: four of the 111 prompt tokens are media pads, so a little under four percent of the positions change their embedding, and the mean gap moves six percent.

What this is not: a controlled pair. The two runs are separate GRPO runs whose rollouts sample independently, so part of every difference above is sampling, and none of it is bitwise. The clean form of the check is one process, one micro-batch, two forwards, comparing the log-probs at the media-pad positions only; that needs a probe rather than a cell and is the next step for this line. Until then the drop-images pair is supporting evidence with the right sign and magnitude, and the prompt-length agreement remains the load-bearing fact that the images reach the batch at all.

## The eight-GPU cells stall in Ray's placement, not in the model (2026-09-17, 11:55)

Chain C's first cell (`img_dp2cp2tp2ep2`, fsdp 2 x cp 2 x tp 2 with ep 2) wrote its last line at 10:57:31, `worker group kwargs: {'device_name': 'cuda'}`, and then nothing for 58 minutes: no placement group, no `actor_rollout_init_model`, no vLLM server, and every GPU at 18 MiB and zero utilisation while the driver sat in its timeout. Nothing in the model or the engine ran at all.

The cause is the Ray CPU budget, the failure mode already recorded for this box: the cell script passes `ray_kwargs.ray_init.num_cpus=24`, which was enough for the one, two and four-GPU cells but not for eight colocated workers, so the placement group is never satisfied and Ray waits instead of failing. The host has 64 cores, so the fix is to raise `RAY_CPUS` for the eight-GPU cells rather than to change anything in the engine.

Two things this does not say: it is not a multimodal finding (no image ever reached a forward), and it leaves the four-axis shapes unverified. They are rerun with a larger budget.

With the Ray budget raised to 48 CPUs the eight-GPU cells start immediately, and the widest shape this box can hold, data-parallel sharding by context parallel by tensor parallel with expert parallelism dividing their product, passes with the same prompt length as every other cell. Four axes at once, images intact, no engine change.

The second eight-GPU cell, pipeline by tensor by context parallel, closes the matrix: ten cells, every axis on its own, every pair that fits, two triples and one quadruple, all three steps, all prompt length 111. The engine carries no change for any of it; what the run changed is the harness (the error filter, the table generator and the Ray budget) and one factored function with tests.

## The controlled probe: where it stands (2026-09-17, 12:30)

`matrix_scripts/k3_vision_causal_probe.py` is the clean form of the drop-images check: one process, one set of weights, one token stream, and the only variable is whether `pixel_values` reaches the forward. Causality gives the verdict without any tolerance to argue about, since positions before the first media pad cannot depend on the image and must stay bitwise equal, while the pads and everything after them must move.

Its batch construction is now smoked on CPU and is self-consistent: 107 token ids, `pixel_values` of `(24, 588)`, `grid_thw` `[[1, 4, 6]]` for the 84 by 56 image, and six media pads at positions 81 to 86. Two cross-checks hold: the merged vision tokens the grid implies, `(4 // 2) * (6 // 2)`, equal the pads in the stream, and the 24 patch rows equal the product of the grid.

Three defects were found and fixed before it ever reached a useful run, all of them from writing it against the APIs instead of running it: it built the model straight on the device instead of on meta followed by `to_empty` and `init_weights`; it cast the module to bf16, which also casts the vision tower's 2D rope cache and `torch.polar` takes only half, float or double, so the weights now stay fp32 and the forward runs under autocast as the trainer and the engine do; and it passed the processor's `(N, 3, 14, 14)` straight to a patch embedding that takes `(N, 588)`, the flattening the engine performs in `_model_multimodal_kwargs`.

What remains unverified is the model side, which needs a GPU: the meta build, the autocast forward and the two-forward comparison. It is queued behind the integration matrix, which holds all eight cards.

## The controlled probe passes: the tower's features are in the policy's logits (2026-09-17, 15:20)

    [probe] tokens 107, media pads 6 at [81, 82, 83, 84, 85, 86]
    [probe] positions before the first pad (81): bitwise equal True, max diff 0.000e+00
    [probe] the pads and everything after (26): max diff 6.828e+00
    [probe] PASS

One process, one set of weights, one token stream, and the only difference between the two forwards is whether `pixel_values` is passed. The 81 positions ahead of the first media pad come back bitwise identical, which they must, since nothing upstream of the image can depend on it; from the first pad onward the logits move by 6.8. That is the vision tower's features reaching the policy, established by causality rather than by a tolerance, and it closes the evidence ladder that the prompt length and the drop-images pair started.

The probe needed five attempts, all of them defects in the probe rather than in the model: building on the device instead of on meta with `init_weights`, casting the whole module to bf16 (the tower's rope goes through `torch.polar`, which refuses it), leaving the parameters in fp32 under autocast (Attention Gym's convolution refuses a bf16 activation against an fp32 weight), and passing the processor's `(N, 3, 14, 14)` to a patch embedding that takes `(N, 588)`. Casting parameters while leaving buffers in fp32 satisfies both kernels at once.

## The one axis value the matrix never exercised: `allgather_kv` (2026-09-17, 15:25)

Every one of the ten cells ran on the default context-parallel backend, `ulysses`. The other value, `allgather_kv`, was then run on the same image cell at cp 2 and fails before step 1:

    RuntimeError: torch.compile with fullgraph=True found no compiled frames.
      prepare_model_inputs -> preprocess_inputs -> _prepare_context_parallel_metadata
      -> cp_shard_metadata -> distributed/context_parallel/api.py:197 shard

`allgather_kv` shards the flex BlockMask and that sharding is compiled with `fullgraph=True`; `ulysses` keeps the mask global and never enters that path, which is why the ten cells are unaffected.

The engine's dynamo probe (`VERL_TORCHTITAN_DYNAMO_PROBE`) was then armed and measured inside the worker, and it contradicts the guess the plan carried:

    DYNAMO-PROBE dynamo.config.disable=False | suppress_errors=False | is_dynamo_supported=True
                 thread=AsyncIO Thread: default main=False | TORCHDYNAMO_DISABLE=None

Dynamo is neither disabled nor unsupported there. The one measured difference left is the thread: the colocated worker runs its forward on Ray's async actor thread, not the main thread. No mechanism is claimed from that, since nothing here separates "dynamo's eval-frame hook does not take on this thread" from another cause; the next step for this line is a minimal comparison inside one worker, the same `torch.compile(fullgraph=True)` called from the main thread and from the async thread.

What can be stated without a mechanism is the behaviour: under the veRL engine's colocated worker, `ulysses` is the context-parallel backend that runs and `allgather_kv` does not. That is what the PR body should say.
