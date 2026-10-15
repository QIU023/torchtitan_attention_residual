# K3 2.8T 入口带图的分辨率上限，titan 的视频 debug 数据（2026-10-15）

## 2.8T 入口是什么
- `scripts/validation/kimi_k3/`（含 `validate_distmuon.py`、`run.sh`）只在 Elfie 的工作分支 `work/kimi-k3-mx-qat-distmuon`（elfiegg/torchtitan，`b7b5eb2b9`，文件来自 `d3b5271d6`，09-30）和我们的栈 `k3_mxqat_stack_20261012` 上。upstream main（`d67ad137e`）没有，她的 draft PR #4791 也没有（31 个文件，脚本只有 checkpoint_conversion）。
- titan 没有满尺寸 K3 训练 recipe：`torchtitan_recipes/models/` 只有 deepseek_v3 / flux / llama3 / muse_glimmer / qwen3；K3 recipe 全是 debugmodel。
- `run.sh full256`：PP8 x DP32 x EP32，seq 2048，micro-batch 2 x 64，1F1B，HybridEP，`recompute-matmuls`（她的 RegionAC），随机 token 手工拼 micro-batch，ViT 不跑。`full128` 是 seq 16，放不下图。
- full256 纯文本还没跑通：10-01 在 QAT 反向 OOM（#4272 comment 5941709055），之后 #4272 没有新结果；README 写 Pending。

## 分辨率上限（逐层）
| 层 | 上限 | 依据 |
|---|---|---|
| 模型 | 无硬上限 | pos-embed 64x64 双线性插值，2D RoPE 现算 |
| 官方预处理 | 65536 patch / 图（16384 token），每边 512 patch（7168 px）；正方形 3584 px | `preprocessor_config.json` |
| titan 数据 | 不放大（`resize_to_navit_patch_grid`）；超过 `max_context_length` 的样本跳过（`mm_datasets.py:411`） | 代码 |
| full256 的 seq 2048 | 每张约 44 x 44 = 1936 token，7744 patch，正方形 1232 px（16:9 约 1652 x 924） | 算术 |
| cc12m-test | 原图 413-840 px，最大 840 x 500（约 2160 patch，540 token） | 本机读 tar |

## 显存（模型估算，来源 10-08 H100 K3 tower 实测表，`CPMM_4380_PREP_2026-10-06.md`）
- 每 patch 激活（扣掉固定 1.7 GiB）：SAC 0.66 MiB，AC 关 0.94 MiB，全量 AC 0.07 MiB。
- PP8 的 1F1B 下 stage 0 同时挂 8 个 micro-batch 的 tower 激活；titan 对 tower 用和 decoder 同一个 AC 配置（`model.py` `policy.apply(self.vision_encoder)`）。
- 每 micro-batch 两张 1232 px（15488 patch）：SAC 每个 10.0 GiB，8 个 80 GiB；全量 AC 合计约 7-9 GiB。
- cc12m-test 原图（每 micro-batch 至多 4320 patch）：SAC 合计约 22 GiB；全量 AC 约 2.3 GiB。
- 余量：她的 memory32（stage 0 代理：11 层前缀，8 个 forward 在途，seq 2048 x 2，无优化器）峰值 174.37 GB，GB200 每卡 186 GB。
- 她的 recompute-matmuls 在 tower 上介于 SAC 和全量 AC 之间，没量过。
- 时间：stage 0 每步编 128 张图（64 x 2）；H100 上 1008 px SAC 每张 149 ms，约 19 s/步只落在 stage 0。

## 视频 debug 数据
- `synthetic-video-text`（`torchtitan_recipes/tests/datasets/synthetic_multimodal.py`，资产 `tests/assets/synthetic_multimodal/video_text/`，#5028 jinsooihm 10-04）：32 行文本配 4 个确定性 uint8 NPY 片段，(帧, 高, 宽) = (2,28,36) (3,32,24) (4,40,30) (5,26,42)，渐变底上移动的色块；`generate.py --check` 校验。
- 只有 `qwen35_debugmodel_video` 用它；integration test 没有格子跑它；CPU 单测覆盖数据路径。
- MM_DATASETS 里真实数据（obelics、cc12m、cc12m-test）都只有图。真实视频文件走 `process_video`（PyAV，默认 2 fps，4-768 帧）。
- 对 K3：patch 14 下这些片段每帧 2-3 patch 一边，2x2 合并后每帧 1 token 左右，时间池化后每段 1-2 token；只覆盖代码路径。

## K3 视频 PR（栈完成后）
- `KimiK3Model.forward` 遇到视频直接 NotImplementedError（`model.py:646`）。
- tower 已有时间维：`_tpool_patch_merger`（2x2 合并后对至多 4 帧取均值，= 官方 `sd2_tpool`），时间 pos-embed（max_num_frames 4 = 官方 `init_pos_emb_time`）；10-08 #4380 的 fp32 检查覆盖了 4 帧 1008 px 视频 item。
- 官方发布的 processor 没有视频分支；config 里有 sample_fps 8、temporal_merge_kernel_size 4、in_patch_limit_each_frame 16384、in_patch_limit_video 655360、timestamp_mode hh:mm:ss.fff。视频 prompt 格式需要报告原文。
