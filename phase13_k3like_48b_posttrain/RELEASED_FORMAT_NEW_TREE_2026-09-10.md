# The released checkpoint format on the new tree (2026-09-10)

Port of the old tree's released key map, quantization scope, official-size configs and vision preprocessing parity onto `k3_int_20260910` (goal item 4). Commit `74711243f` in the detached worktree `/tmp/wt_moonep` (cherry-picked onto the branch after the CP-mm commit).

## What the new tree already had

- `KimiK3StateDictAdapter` maps the released multimodal spelling (`language_model.model.*`, `vision_tower.*`, `mm_projector.*`), MLA and KDA by layer type, expert stacking, the fused vision `wqkv`, and the layer-0 attention residual placeholders. The old `hf_key_map.py` adds only the text-only spelling (`model.*`, for the Kimi-Linear-48B graft) and config export.
- The "Kimi-K3" flavor is the official topology field for field (93 layers, full attention at 4, 8, ..., 92 and 93, block size 12, MLA 128/64/128 with q_lora 1536 and kv_lora 512, KDA 96 x 128, 896 experts top-16, latent 3584, expert hidden 3072, dense 33792, vision 27 x 1024 with 64 x 64 position tables and 4 frames). The only difference is the context length the flavor registers (262144) against the released `max_position_embeddings` 1048576.
- The shared NaViT resize (`torchtitan/hf_datasets/multimodal/utils/image.py`, `resize_to_navit_patch_grid`) is the released `navit_resize_image` formula; K3's dataloader config uses it with the raster patch order.
- MXFP4 QAT is a config-tree converter whose scope is every `GroupedExperts.Config`.

## What moved

- Flavors `report_arch` (13 layers, width 256, full attention 4/8/12/13, block 7, MLA 32/16/32, KDA 4 x 128, 8 experts top-2, latent 256, expert hidden 224, dense 896, tower 4 x 256) and `k3mini` (21 layers, width 512, block 12, released head dims, same tower). `report_arch` is the topology of the debug-size checkpoint Moonshot ships in the released layout (`/workspace/k3qat_mm_hf`: config, one safetensors shard, processor code, tokenizer).
- Adapter: `A_log` travels as the released `[1, 1, heads, 1]` and flattens on import (dt_bias already did the equivalent); `_check_not_packed` refuses `weight_packed` / `weight_scale` / fp8 / uint8 tensors; `get_hf_storage_reader(from_quantized=True)` returns torch's `QuantizedHuggingFaceStorageReader`, so the released MXFP4 experts (packed as `.weight_packed` plus `.weight_scale`, group 32, E2M1 with E8M0 scales) arrive dequantized. The old `packed_mxfp4.py` dequantizer is not ported: torch's reader implements the same format.
- Tests (`tests/unit_tests/cpu/`, all skip when the artifacts are absent, paths overridable by `KIMI_K3_RELEASED_DEBUG_DIR` and `KIMI_K3_RELEASED_INDEX`):
  - `test_kimi_k3_released_checkpoint.py`: the released-layout shard loads into `report_arch` with every key on a parameter of the right shape (412 parameters, 658 released keys, 0 missing / 0 extra / 0 shape mismatches) and exports back bitwise; the twin's dimensions match the artifact's config; packed keys are refused.
  - `test_kimi_k3_quant_scope.py`: the released 2.8T index (`official_k3/reference/model.safetensors.index.json`) quantizes exactly the routed experts (`targets: ["Linear"]` with the six ignore regexes resolves to `block_sparse_moe.experts.N.w{1,2,3}` and nothing else), and the QAT converter reaches exactly the routed-expert modules on the twin.
  - `test_kimi_k3_vision_preprocess_parity.py`: the shared resize against the released `media_utils.navit_resize_image` at the released budgets (65536 patches, 512 per side, patch 14, merge 2) on ten image sizes, exact on resized size, padding and token count.

## Not ported, and why

- The Kimi-Linear-48B graft (`_gated` / `_gated_lora` flavors, alpha-gated attention residual reads, the text-only key spelling): the released Kimi-Linear-48B-A3B is not K3-shaped (SiLU, ungated MLA, no latent MoE, one shared expert, routed scaling 2.446, no q_lora), and the new tree's K3 modules implement K3 only. Grafting from it needs a second architecture variant in the K3 folder or a Kimi-Linear model of its own; a design decision, not a port.
- Config export to the released `config.json` (`titan_config_to_official`): no consumer on the new tree yet (vLLM export goes through to_hf weights plus the released config).
- Video preprocessing (`in_patch_limit_each_frame`, `pack_video`): the K3 model on the new tree rejects videos (`pixel_values_videos` raises), so the per-frame budget has nothing to feed.
- The old scaling-law sweep sizes (194m to 528m): not K3-shaped either.
