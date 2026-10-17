# titan：K3 视频训练（按 4 帧分段、带时间戳、和图片共用一条流）（2026-10-15）


- **10-17（用户："改"）：** `k3_video` = `75b934a8f`，两个提交，已 rebase 到 upstream main `8a3f5bcaa`。旧 head `33d1630c0` 备份在 `backup/k3_video_pre_20261017`。
  - 新提交 `75b934a8f`：`KimiK3VisionEncoder.compute_position_embeddings` 遇到帧数超过 `max_num_frames` 的 item 就报错（只读 host 上已有的 grids，不多一次同步）；分块函数的 docstring 写明 `fps` 和 `frames_per_chunk`；对拍脚本给 max_pixels 和 min_pixels 起了常量名，文件说明写清 `max_num_frames` 是 encoder 能接受的最长 item。
  - 正文：用分号连着两件事的条目都拆开了；Design 写清 `frames_per_chunk` 和 `max_num_frames` 的关系，Refusals 加了一条；5060 训练表改成一句覆盖说明（数值规则：数只放 H100，5060 只说跑没跑通）。
  - 测试（rebase 后）：CPU 22 passed，GPU `test_kimi_k3.py` 6 passed（视频用例里加了帧数检查）。
  - 10-17 用户定了："不改图片路径，只改视频"。提交 `659c72b74` 加了 `chunk_header`（kit 的 recipe 传 `video<|media_content|>`）；22 格要在这个提交上重跑后放进正文（用户："视频PR是新change，不存在数值问题，直接把5060的22种布局放进去"）。当时的事实：
    - 发布的 K3 processor（本机 `/workspace/k3qat_mm_hf/kimi_k3_vision_processing.py`）把图片写成 `<|media_begin|>image WxH<|media_content|><|media_pad|><|media_end|>`，没有视频分支；视频的 `video<|media_content|>` 来自 K2.5 的 processor，本 PR 的对拍脚本 HF 那侧已经这么写。
    - titan 的图片和视频块都只写 `<|media_begin|><|media_pad|>…<|media_end|>`。
    - 测试用 tokenizer（`tests/assets/tokenizer`）和本机那份发布布局的 tokenizer 都没有 `<|media_content|>` 这个特殊 token，会按普通文本切开。
  - 我的建议：
    - 视频的包装放进本 PR：只改 `_process_chunked_video_text_sample` 传给 `insert_vision_placeholders` 的开始串，不碰共享代码，没有 golden 要动，训练数据和对拍的 prompt 就一致了。
    - 图片的包装另开 PR：它会改每个 K3 图片样本的 token，K3 的 B200 golden（逐位）要重新生成；放置占位符的函数是所有 VLM 共用的，要给 K3 单独加开关。

## 状态（不粘贴）

- **10-15 CPU 会话复查（用户："拉取 检查栈分支和视频pr分支"）：**
  - 分支 `k3_video` = `33d1630c0`，一个提交，基于 `d51e4b3c6`；main 已到 `18f5a6082`（多 19 个提交），合并没有冲突。
  - 逐行读了 diff，代码没有问题。
    - `format_video_timestamp` 和 K2.5 发布的 `media_utils.timestamp_as_str` 写法一致（同样用 `int((t % 1) * 1000)` 截断毫秒），从 HF 上取原文件对照过。
    - 时间戳的取值有一点不同：K2.5 用每段首帧在原视频里的帧号除以原视频 fps，本 PR 用 `index * frames_per_chunk / fps` 这个名义值。原视频 fps 不是采样 fps 的整数倍时，两者会差几毫秒。函数拿到的是已经采样好的帧，算不出原始帧号；docstring 可以写明 frames 是按 `fps` 采样的。可选。
    - `_video_chunks` 里 `max_pixels=16777216`、`min_pixels=65536` 直接写成了数字，没有像其他常量那样起名。可选。
  - body 有两点和规则或用户的决定对不上：
    1. Results 的训练表是 RTX 5060 Ti 上测的，而用户的第 3 条决定是"数值标 Pending (H100)"。
    2. 同一条数据流里第 1 步的 loss 不一致（dp1 那组：8.09021，对 tp2 × ep2 的 8.09039、cp2 的 8.09048、四轴那格的 8.09016），表里没有定位。按数值规则，这些要么定位（TP 和 CP 的规约顺序），要么不进表。
    - 建议这一段改成一句 "all 22 layouts finish 10 steps"，或者整体标 Pending (H100)，由用户定。
  - Design 里 "A dataset uses the processor ...; this PR adds no recipe." 一条用分号连着两件事，可以拆开。

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
- **10-16 重新验证（`33d1630c0`，用户："本机能验证栈分支和视频pr分支的结果吗？视频pr body和diff重新审核"；栈的部分按用户的话取消："不需要再次验证栈"）：**
  - GPU 单测 `test_kimi_k3.py` 6 passed。
  - 对拍（本机 shim）：
    - 图片 KL 7.6e-6，8 帧 6.0e-6，5 帧 1.0e-5，top-1 一致，top-5 100%。
    - 和 10-15 的 4.8e-6、3.7e-6、6.9e-6 有起伏，是重新冷编译后 autotune 选到的 kernel 不同。正文本来就是 Pending (H100)，不受影响。
  - 22 格矩阵（`verify_1016.sh`，在 10-15 那轮缓存和 seed 的副本上跑，recipe 用 kit 里的那份）：22/22 rc 0，每一格每一步的 loss 和 grad norm 都和 10-15 那轮相同（`compare_runs.py`，日志精度）。正文的表不用改。
  - upstream main 比基底多 21 个提交（到 `95788b053`），都没碰本 PR 的文件和多模态、K3 相关代码，`merge-tree` 无冲突。
  - 重审 diff 和正文，代码没发现错误。正文漏看了两点，10-15 CPU 会话的复查（上面那一节）指出了：训练表是 5060 上的数，和用户的 Pending (H100) 决定矛盾；同一份数据的各格之间第 1 步有差，没有定位。这两点成立，正文要改。三条建议等用户定：
    1. K3 侧检查每段帧数不超过 `max_num_frames`。现在超过时 tower 照常编码，不报错。
    2. Design 的 Chunks 写成 "up to `frames_per_chunk` frames (K3 sets it to `max_num_frames`)"。
    3. 正文明说视频段没有 K2.5 的 `video<|media_content|>` 头，和图片块一样。这和媒体块写法是否另开 PR 是同一件事。
- **原来待用户决定的事（已答）：**
  - `kimi_k3_debugmodel_video` 这个 recipe 留不留？要不要配 CI 格子？项目规则是 recipe 只给 CI 实际跑的形状，加之前要问。Qwen3.5 有同样的视频 recipe，但也没有 CI 格子。
  - 要不要把视频对拍做成 `numerical_tests_kimi_k3.py` 的 `--num_frames` 选项放进 PR，让 reviewer 能复现。
  - 对拍的数值是在 5060 上量的。按规则，PR 里的数值要用 H100 量，否则标 Pending (H100)。
- **怎么开：** `gh` 不在本机，PR 由用户在 https://github.com/pytorch/torchtitan/compare/main...QIU023:torchtitan:k3_video?expand=1 开。

标题：`[Kimi K3] Train on video in timestamped chunks`

--- PASTE BEGIN ---

## Summary

Kimi K3 rejects video inputs on main. This PR trains it on video by turning each clip into timestamped chunks of up to four frames that join the image stream in document order.

- `torchtitan/components/data/multimodal/mm_datasets.py`: `_process_chunked_video_text_sample` turns sampled frames and a caption into chunk items in `pixel_values`, each block after its start time and opened by `chunk_header`.
- `torchtitan/components/data/multimodal/utils/video.py`: `format_video_timestamp` writes `hh:mm:ss.fff`.
- `torchtitan/models/kimi_k3/model.py`: the `pixel_values_videos` refusal now points to chunks.
- `torchtitan/models/kimi_k3/vision_encoder.py`: the encoder refuses an item longer than `max_num_frames` frames.
- `scripts/checkpoint_conversion/numerical_tests_kimi_k3.py`: `--num_frames N` runs the parity test on an N-frame clip.

## Design

- Stream:
  - K3 uses `<|media_pad|>` as the placeholder for both images and videos.
  - A separate video stream cannot be matched to its placeholder runs.
  - `kimi_k2_7/model.py` notes the same and points to one document-ordered stream.
  - Chunks therefore go into `pixel_values`, in document order with the images.
- Chunks:
  - K3's vision encoder pools each item over time and adds the temporal position embedding.
  - `max_num_frames` (4 for K3) is the longest item the encoder takes.
  - `_process_chunked_video_text_sample` cuts chunks of `frames_per_chunk` frames, which a recipe sets to `max_num_frames`.
  - The last chunk may be shorter.
  - Each chunk has `(h/2)·(w/2)` placeholder tokens.
  - Each chunk counts as one entry against `max_images_per_microbatch`.
- Spelling:
  - Each chunk's placeholder run follows its start time as `hh:mm:ss.fff`, as the released Kimi K2.5 processor writes it.
  - The released K3 preprocessor config sets the same 8 fps, four-frame merge and timestamp mode.
  - The chunk block reuses the image block's start and end tokens.
  - `chunk_header` follows the start token.
  - The released Kimi K2.5 processor writes `video<|media_content|>` there.
  - Images keep their current spelling.
- Refusals:
  - A tokenizer whose video placeholder differs from the image one.
  - A temporal patch size above 1.
  - `pixel_values_videos` in `KimiK3Model`.
  - An item longer than `max_num_frames` frames, in `KimiK3VisionEncoder`.
- Parity test:
  - The released processor has no video branch, so both sides take the chunk patches built here.
  - The HF prompt spells each chunk as the released Kimi K2.5 processor does.
  - The comparison covers the vision features, the routing and the last-token logits, as for the image.
- Wiring:
  - A dataset uses the processor through `VisionProcessor.Config(sample_processor=...)`.
  - This PR adds no recipe.

## Test plan

```
pytest tests/unit_tests/cpu/components/data/test_mm_dataset_preprocessing.py  # 22 passed, 4 new
pytest tests/unit_tests/gpu/models/kimi_k3/test_kimi_k3.py                     # 7 passed, 2 new
python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3                 # image, unchanged
python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3 --num_frames 8  # two 4-frame chunks
python -m scripts.checkpoint_conversion.numerical_tests_kimi_k3 --num_frames 5  # chunks of 4 and 1 frames
```

## Results

Parity with the released modeling code, debug checkpoint, bf16:

| Input | Vision features cos / max diff | Router agreement | Last-token KL | Top-1 |
|---|---|---|---|---|
| image |  |  |  |  |
| 8 frames, two 4-frame chunks |  |  |  |  |
| 5 frames, chunks of 4 and 1 |  |  |  |  |

Training on `synthetic-video-text` through `_process_chunked_video_text_sample`, debug model, AdamW, one seed checkpoint, 8x RTX 5060 Ti, loss / grad norm:

| Layout | Step 1 | Step 5 | Step 10 |
|---|---|---|---|
| dp1 | 8.17278 / 3.4375 | 1.53622 / 5.7188 | 0.52857 / 1.3984 |
| pp2 | 8.17278 / 3.4375 | 1.53539 / 5.7188 | 0.53782 / 1.4219 |
| pp2×vpp2 |  |  |  |
| pp4 | 8.17278 / 3.4375 | 1.53501 / 5.6875 | 0.52832 / 1.3984 |
| pp8 | 8.17278 / 3.4219 | 1.53679 / 5.7500 | 0.53094 / 1.3828 |
| tp2×ep2 | 8.17287 / 3.4062 | 1.53710 / 5.6875 | 0.53408 / 1.4219 |
| tp4×ep4 | 8.17221 / 3.4062 | 1.54265 / 5.6250 | 0.53281 / 1.4141 |
| cp2 |  |  |  |
| cp2, type-checked |  |  |  |
| ep2×tp2×pp2×cp2 |  |  |  |
| cp4, seq 512 |  |  |  |
| fsdp2 | 8.16997 / 3.3906 | 1.51793 / 5.6562 | 0.53946 / 1.4141 |
| ep2×fsdp2 | 8.16997 / 3.4062 | 1.51872 / 5.5938 | 0.55638 / 1.4531 |
| ep2×fsdp2×tp2, type-checked |  |  |  |
| ep4×fsdp2×tp2×pp2 | 8.17032 / 3.3906 | 1.51841 / 5.6250 | 0.55667 / 1.4609 |
| ep2×fsdp2×tp2×pp2 | 8.17032 / 3.3906 | 1.51513 / 5.6250 | 0.55234 / 1.4453 |
| fsdp2×pp2×cp2 | 8.16992 / 3.3906 | 1.51577 / 5.5938 | 0.54904 / 1.4531 |
| ep2×fsdp2×pp2×cp2 | 8.16992 / 3.3906 | 1.51700 / 5.5938 | 0.55339 / 1.4688 |
| ep4×fsdp2×tp2×cp2 | 8.16974 / 3.4062 | 1.51503 / 5.5938 | 0.54493 / 1.4531 |
| ep2×fsdp2×tp2×cp2 |  |  |  |
| ep8×fsdp8 | 8.16789 / 3.4062 | 1.53114 / 5.6875 | 0.52807 / 1.3438 |
| ep2×fsdp2, cc12m-test images mixed in |  |  |  |

- Rows from dp1 and from dp2 read different samples.
- The set holds 32 captions over 4 clips, so the model memorises it within 10 steps.

--- PASTE END ---
