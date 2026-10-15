# titan：K3 视频训练（按 4 帧分段、带时间戳、和图片共用一条流）（2026-10-15）

## 状态（不粘贴）

- **来由：** 用户 10-15："现在直接开 从最新的mainline开始"。
- **分支：** fork `QIU023/torchtitan` 的 `k3_video` = `33d1630c0`（10-15 按用户的 4 点修改后 amend，旧提交是 `93aaf9fdf`），基于 upstream main `d51e4b3c6`（10-14 23:15 PDT），一个提交，作者 QIU023，没有 trailer。6 个文件 +323 / −51。
  - recipe 已拿掉，两个 recipe 文件和 main 相同；本地复跑用 kit 里 `k3v_cells.py` 自带的同名配置。
  - 新增的 docstring 都是一行，另有 4 个 `pyrefly: ignore`，照文件里原有写法。
- **为什么不照 Qwen3.5 用 `pixel_values_videos`：** K3（和 K2.5）的 `image_token` 和 `video_token` 都是 `<|media_pad|>`。图片和视频如果分两条流，模型无法按占位 id 区分各个 run 属于哪条流。K2.7 在 titan 里因此只支持单一模态的 batch，并留了 TODO："when image_id == video_id, emit one document-ordered vision stream"。本 PR 就是按这个 TODO 做的：数据侧把视频段放进 `pixel_values`。
- **格式出处：** K3 发布的 processor 没有视频分支，但 `preprocessor_config.json` 写了 `sample_fps 8.0`、`temporal_merge_kernel_size 4`、`timestamp_mode hh:mm:ss.fff`。K2.5 发布的 `kimi_k25_vision_processing.py` 的写法是每 4 帧一段，每段为 `"{ts}<|media_begin|>video<|media_content|><|media_pad|><|media_end|>"`，`ts` 取该段首帧的时间。
  - titan 的图片块原本就只写 `vision_start + pads + vision_end`，没有官方的 `image WxH<|media_content|>`。视频段沿用同一写法，前面加时间戳，所以也没有 `video<|media_content|>`。两者要不要都改成官方写法，是另一个问题，留给用户决定。
- **K3 报告：** §2.4 说 MoonViT-V2 对图片和视频共用参数，"attention is factorized into intra-frame spatial and inter-frame temporal passes"。但发布的代码在一段之内对 t×h×w 做整段注意力（cu_seqlens = t·h·w），titan 的实现也是这样。对拍以发布代码为准。
- **本机验证（8 x RTX 5060 Ti，venv_1006；main 的固定依赖装在 `scratchpad/main_pins_1015`：spmd_types a1b86de、torch_remat 624f19c、torch_checkpointing dcbb3a5、attn-gym 0.0.17）：**
  - 本机没有 FA4。K3 默认的 varlen 后端在 sm120 上要么缺 `flash_attn`，要么换成 torch flash 后不支持 MLA 的 v 维。所以全部用 flex 后端，和 CI 的 K3 多模态格子一样。图片 recipe 在本机同样受这个限制。
  - 单测：
    - `test_mm_dataset_preprocessing.py` 22 passed（新增 4 个）。
    - `gpu/models/kimi_k3/test_kimi_k3.py` 6 passed（新增 1 个，单卡）。
    - CPU 的 data、kimi_k3、qwen3_5 三组：219 passed、2 skipped，另有 3 个 failed。这 3 个是 Qwen3.5 MRoPE 测试的 CPU inductor 编译错误，在干净的 main 上第一次失败、第二次通过，是本机的问题。
    - 新的模型测试最初写在 CPU 上，但本机 torch nightly 的 CPU flex 编译有 bug（`cur_qSplitSize8`），所以放到了 GPU。
  - recipe 的 batch（`probe_loader.py`）：每段 t ≤ 4，5 帧的视频切成 4 帧 + 1 帧两段，每段的占位长度等于 (h/2)(w/2)，时间戳在段前，`pixel_values_videos` 为空。
  - 和发布代码对拍（`parity_video.py`）：
    - 用 `numerical_tests_kimi_k3.py` 的固定 revision 和缩小的 debug 配置，权重严格转换，bf16，两边喂同一份分段 patch。
    - fla-core 0.5.2 和本机的 Triton 不兼容，所以把 fla 换成空模块，只比视觉路径（tower + projector + scatter）。decoder 和图片走的是同一条路。
    - 结果见正文的表。
  - 矩阵（`run_k3v.sh`，同一个 seed checkpoint，flex，AdamW，10 步，10-14 23:47 到 10-15 00:29 PDT）：
    - 22 格全部 rc 0，跑满 10 步，每个 rank 都加载了 seed：18 格多模态矩阵的形状，加类型检查的 ep2 × fsdp2 × tp2 和 cp2、pp2 × vpp2、图片加视频混合的 ep2 × fsdp2。
    - 第 1 步按数据流分组：
      - dp1 那组：dp1、pp2、pp2 × vpp2、pp4、pp8、tp4 × ep4 都是 8.09021；tp2 × ep2 是 8.09039；cp2 开不开类型检查都是 8.09048；ep2 × tp2 × pp2 × cp2 是 8.09016。
      - cp4（seq 512，每步的样本不同）是 8.08564。
      - dp2 那组：fsdp2 和 ep2 × fsdp2 是 8.09241；带 TP 和 PP 的格子是 8.09227；fsdp2 × pp2 × cp2 那两格是 8.09250；fsdp2 × tp2 × cp2 那两格是 8.09211。
      - ep8 × fsdp8 是 8.08933。
    - 第 10 步在 0.408 到 0.439 之间。混合那格第 10 步是 2.811，因为图片 caption 还没背下来。
    - 汇总脚本是 `kit_2026-10-15/summarize.py`。
- **diff 审读（10-15，逐行读了 `git diff d51e4b3c6 93aaf9fdf`）：**
  - 新增 4 个 def：
    - `_process_chunked_video_text_sample`：和 `_process_video_text_sample` 并列，复用 `process_video`、`calculate_vision_tokens`、`insert_vision_placeholders`、`_tokenize_mm_text`，没有现成的分段版本。
    - `format_video_timestamp`：titan 里没有时间戳函数（`git grep timestamp torchtitan` 只有日志和 graph_trainer 的脚本）。
    - `kimi_k3_debugmodel_video`（recipe）和测试 helper。
  - 没有 logbook 路径，没有实验数字进代码，docstring 各一行。
  - recipe 里的 `fps=8.0` 是 K3 发布的预处理配置的采样率，titan 里没有可以引用的工具函数，正文的 Design 有说明。
  - 正文里没有破折号，也没有 we、our、us。
- **10-15 用户的决定：**
  1. recipe 从 PR 拿掉。
  2. 视频对拍做成 `numerical_tests_kimi_k3.py` 的选项放进 PR。
  3. 数值标 Pending (H100)。
  4. 问媒体块写法是什么意思，已解释，等用户定要不要另开 PR。
- **脚本选项本机跑通（5060 Ti；titan 侧用 `kit/flexshim` 强制 flex，因为本机没有 FA4；参考侧用 fla 0.6.0（git 主线），因为钉的 0.5.2 在 Triton 3.9 上 import 失败）：**

  | 输入 | tower cos / 最大差 | 路由一致 | logits KL | top-1 / top-5 |
  |---|---|---|---|---|
  | 图片 336 px（不加选项） | 0.999987 / 3.1e-2 | 5221/5248 | 4.8e-6 | 一致 / 100% |
  | 8 帧 336 px（4+4） | 0.999992 / 3.1e-2 | 10210/10272 | 3.7e-6 | 一致 / 100% |
  | 5 帧 336 px（4+1） | 0.999989 / 3.1e-2 | 10208/10272 | 6.9e-6 | 一致 / 100% |
  | 8 帧 448 px（4+4） | 0.999982 / 3.1e-2 | 17349/17440 | 6.4e-6 | 一致 / 100% |

  这些数字不进正文（Pending (H100)）。日志在 `scratchpad/k3v_run/parity2/`。
- **改动后重跑：** data 的 CPU 测试 197 passed（偶尔有 1 个 `test_packed_vision` 的 CPU flex 编译错误，3 次中 1 次，代码 main 上就有），K3 GPU 测试 6 passed，脚本的 pyrefly 0 个错误（main 也是 0），flake8 只有 main 原有的 B043。
- **原来待用户决定的事（已答）：**
  - `kimi_k3_debugmodel_video` 这个 recipe 留不留？要不要配 CI 格子？项目规则是 recipe 只给 CI 实际跑的形状，加之前要问。Qwen3.5 有同样的视频 recipe，但也没有 CI 格子。
  - 要不要把视频对拍做成 `numerical_tests_kimi_k3.py` 的 `--num_frames` 选项放进 PR，让 reviewer 能复现。
  - 对拍的数值是在 5060 上量的。按规则，PR 里的数值要用 H100 量，否则标 Pending (H100)。
- **怎么开：** `gh` 不在本机，PR 由用户在 https://github.com/pytorch/torchtitan/compare/main...QIU023:torchtitan:k3_video?expand=1 开。

标题：`[Kimi K3] Train on video in timestamped chunks`

--- PASTE BEGIN ---

## Summary

Kimi K3 rejects video inputs on main; this PR trains it on video by turning each clip into timestamped chunks of up to four frames that join the image stream in document order.

- `torchtitan/components/data/multimodal/mm_datasets.py`: `_process_chunked_video_text_sample` turns sampled frames and a caption into chunk items in `pixel_values`, each placeholder run after its start time.
- `torchtitan/components/data/multimodal/utils/video.py`: `format_video_timestamp` writes `hh:mm:ss.fff`.
- `torchtitan/models/kimi_k3/model.py`: the `pixel_values_videos` refusal now points to chunks.
- `scripts/checkpoint_conversion/numerical_tests_kimi_k3.py`: `--num_frames N` runs the parity test on an N-frame clip.

## Design

- Stream:
  - K3 uses `<|media_pad|>` as the placeholder for both images and videos.
  - A separate video stream cannot be matched to its placeholder runs; `kimi_k2_7/model.py` notes the same and points to one document-ordered stream.
  - Chunks therefore go into `pixel_values`, in document order with the images.
- Chunks:
  - K3's vision encoder pools each item over time and adds the temporal position embedding.
  - A chunk is one item of up to `max_num_frames` frames with `(h/2)·(w/2)` placeholder tokens.
  - Each chunk counts as one entry against `max_images_per_microbatch`.
- Spelling:
  - Each chunk's placeholder run follows its start time as `hh:mm:ss.fff`, as the released Kimi K2.5 processor writes it.
  - The released K3 preprocessor config sets the same 8 fps, four-frame merge and timestamp mode.
  - The chunk block reuses the image block's start and end tokens.
- Refusals:
  - A tokenizer whose video placeholder differs from the image one.
  - A temporal patch size above 1.
  - `pixel_values_videos` in `KimiK3Model`.
- Parity test:
  - The released processor has no video branch, so both sides take the chunk patches built here.
  - The HF prompt spells each chunk as the released Kimi K2.5 processor does.
  - The comparison covers the vision features, the routing and the last-token logits, as for the image.
- A dataset uses the processor through `VisionProcessor.Config(sample_processor=...)`; this PR adds no recipe.

## Test plan

```
pytest tests/unit_tests/cpu/components/data/test_mm_dataset_preprocessing.py  # 22 passed, 4 new
pytest tests/unit_tests/gpu/models/kimi_k3/test_kimi_k3.py                     # 6 passed, 1 new
python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3                 # image, unchanged
python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3 --num_frames 8  # two 4-frame chunks
python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3 --num_frames 5  # chunks of 4 and 1 frames
```

## Results

Parity with the released modeling code (vision feature cos and max diff, router agreement, last-token KL and top-1): Pending (H100).

Training with a local recipe that feeds `synthetic-video-text` through `_process_chunked_video_text_sample` (debug model, flex attention, AdamW, one seed checkpoint, RTX 5060 Ti): all 22 layouts finish 10 steps; the set (32 captions over 4 clips) is memorised within them, so only step 1 is shown.

| Data | Layouts | Step 1 |
|---|---|---|
| dp1 | dp1, pp2, pp2×vpp2, pp4, pp8, tp4×ep4 | 8.09021 |
| dp1 | tp2×ep2 / cp2, type-checked or not / ep2×tp2×pp2×cp2 | 8.09039 / 8.09048 / 8.09016 |
| dp2 | fsdp2, ep2×fsdp2 | 8.09241 |
| dp2 | ep4×fsdp2×tp2×pp2, ep2×fsdp2×tp2×pp2, ep2×fsdp2×tp2 (type-checked) | 8.09227 |
| dp2 | fsdp2×pp2×cp2, ep2×fsdp2×pp2×cp2 / ep4×fsdp2×tp2×cp2, ep2×fsdp2×tp2×cp2 | 8.09250 / 8.09211 |
| dp8 | ep8×fsdp8 | 8.08933 |
| dp1, seq 512 | cp4 | 8.08564 |
| dp2, cc12m-test and video mixed | ep2×fsdp2 | 8.08846 |

--- PASTE END ---
