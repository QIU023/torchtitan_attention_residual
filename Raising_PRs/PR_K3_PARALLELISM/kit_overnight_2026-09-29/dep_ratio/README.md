# DEP 文本 / 视觉计算比例（overnight 09-29 目标二）

结论和表在 `../../DEP_RATIO_2026-09-29.md`。这里是套件和命令。所有文件都是本地的，不提交到 torchtitan。

## 文件

- `build_dataset.py`：从 jackyhate/text-to-image-2M 的 `data_1024_10K/data_000000.tar`（MIT，5.07 GB，一万张 1024 × 1024 的 FLUX 图，每张带 prompt）打本地数据集。每个样本存 4 张不同的原图（`img0.jpg` 到 `img3.jpg`，不重新编码）、一句短 caption（prompt 的第一句，最多 200 字符）和 json（`n_images`，按固定种子从 0 到 4 抽，概率 0.15 / 0.35 / 0.25 / 0.15 / 0.10）。2500 个样本，5 个 shard，4.7 GB，在 `/workspace/dep_data/t2i1024_k4/`（不进 git），`MANIFEST.json` 记了每个 shard 的字节数和 sha256。
- `dep_ratio_local.py`：DEP 树 `a03f74981` 的本地 recipe。每张图 resize 成固定正方形（`DEPR_RES` = 224 / 448 / 768 / 1024 → 边长 224 / 448 / 756 / 1008，会放大；K3 recipe 的 `resize_to_navit_patch_grid` 只缩不放、一张图最多 256 个 patch），`max_patches` 至少 5184、每边至少 72；样本用前 n 张图，n 再按 `DEPR_SEQ` 放得下的数截断；布局照 `kit_h100_2026-09-29/dep/dep4.py`（pp2 × vpp4 × tp2 × ep2，不开 FSDP），`dep_off` / `dep_k25` / `dep_bubble`，`w_*` 按 `DEPW_DIM` 放宽度。设了 `DEPN_MEM_OUT` 时导入 `dep4` 记每个 rank 每步的峰值。
- `check_batches.py`：CPU 上每种（分辨率，seq）抽 24 个 micro-batch，打印占用的位置数、loss token 数、每个 micro-batch 的图数、每张图的 patch 数。输出 `check_batches.out.txt`。
- `flop_ratio.py`：在 meta 上建发布版 Kimi-K3 和放宽到 6144 的 debug model，按参数量加注意力二次项推算"一张图的编码前向 / 一个文本 stage 的前向"。输出 `flop_ratio.out.txt`。
- `microbench_ratio.py`：单卡实测同一个比例（编码前向、编码前向加反向，对 pp2 × vpp4 切分下第 1 到 6 个 stage 的平均），bf16，随机权重，预热 3 次、测 10 次取中位数。只在 CPU 上核对过塔的前向和注意力元数据的构造，文本层要 CUDA（KDA）。
- `run_ratio_5060.sh`：给主会话排队的 5060 作业：先跑 microbench（dim 1024、2048 × seq 1024、2048，单卡），再用 4 卡在三档数据上各冒烟 DEP 关 / K2.5 / bubble 10 步（dim 1024），打印每格的 `vision_dep` 计划行和 rc。
- `run_ratio_h100.sh`：下次 H100 的格子（三档 × 三种配置的计时和 trace，外加 L2 的 20 步数值）。

## 5060 上的命令（主会话排队，GPU 一次一个作业）

```bash
bash Raising_PRs/PR_K3_PARALLELISM/kit_overnight_2026-09-29/dep_ratio/run_ratio_5060.sh
# 结果在 $S/dep_ratio/（progress.txt、microbench_d*_s*.txt、smoke_*/run.log）
```

预计时间：microbench 四组各 3 到 4 分钟（含 flex 和 KDA 的编译），约 15 分钟；冒烟 9 格各约 2 分钟，约 20 分钟。合计约 35 分钟。

## 下次 H100

1. 数据：在机器上从 HF 重下同一个 shard 再打包最快（下载约 2 分钟，打包约 1 分钟，sha256 应与 `MANIFEST.json` 一致）：
   `python -c "from huggingface_hub import hf_hub_download as d; d('jackyhate/text-to-image-2M','data_1024_10K/data_000000.tar',repo_type='dataset',local_dir='/root/dep_data/raw')"`，
   `python build_dataset.py /root/dep_data/raw/data_1024_10K/data_000000.tar /root/dep_data/t2i1024_k4`。
2. 套件拷到 `~/kit/dep_ratio/`，`kit_h100_2026-09-29/dep/` 拷到 `~/kit/dep/`（`dep4.py`、`analyze_trace.py`、`ana_dep4.py`、`steps.py`）。
3. `nohup bash ~/kit/dep_ratio/run_ratio_h100.sh > ~/mep/dep_ratio.log 2>&1 &`，约 70 分钟（每档约 20 分钟，数值约 6 分钟）。
