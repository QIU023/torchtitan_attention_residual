# Rebase 到 4025 新 head `ded1bf109`(2026-08-23)

基点从 `dee45e357` 移到 `ded1bf109`。**这次上游改动对我们不是 non-blocking。**

## 上游改了什么

7 个 commit,其中一个是结构性的:**"share the MoonViT tower between k2.7 and k3"**。

| 文件 | 变化 |
|---|---|
| `kimi_k3/vision_encoder.py` | **935 -> 56 行** |
| `kimi_k2_7/vision_encoder.py` | +49 |
| `common/vision_encoder.py` | +7 |
| `tests/unit_tests/test_kimi_k3.py` | -170(挪进 CI) |

`KimiK3VisionEncoder` 现在是 `kimi_k2_7.vision_encoder.MoonViTEncoder` 的**子类**,
k3 侧只剩一个 `KimiK3VisionProjector` 和一个补字段的 Config。
`VisionRotaryEmbedding2D` 也改从 k2.7 引。

其余六个是清理:`probs_TN -> probs_T1N`、只留 bf16、README 澄清、删两个测试。

## 对我们的影响

24 个 commit 里 **20 个正常 rebase,只有一处冲突**(import 块)。

**4 个 dynamic CP 的 commit 无法机械 rebase** —— 它们改的代码已经不在这个文件里了。
已跳过,并按新结构重新落一次:

| 原来 | 现在 |
|---|---|
| 226 行直接写在 935 行的 `vision_encoder.py` 里 | **原样搬进 56 行的 k3 文件** |
| 分片逻辑改 `forward` 本体 | **`forward` 的 override**,无 plan 时 `super().forward()` |
| gather-KV 改注意力 | **`VisionAttention` 的子类**,由 k3 自己的 config 选中 |
| — | **k2.7 的共享塔一行未动** |

**dynamic CP 留在 k3**,不上移到共享塔:它是 K3 report 5.2.3 描述的特性,k2.5 用不到。
新结构反而让这件事更容易做对。

## 门的设计,这一轮全部按原树对齐

用户三次指正后的最终形态:

1. **格子清单 = 老 58 格剔除 TP**,不是另起炉灶。参数直接取自
   `run13_flav.sh` / `run_maxdeg.sh`。剔掉 6 个含 tp 的,留 10 格 x 3 臂。
2. **一份种子 ckpt 覆盖全部 30 格**。实测支持:text 的 703 个 key 是 mm 的 773 的
   **严格子集**(少的 70 个全是视觉塔);lora 的 809 个**包含全部 773 个**,
   多出的 36 个是适配器,本来就该各自初始化。加载端是 `strict=False`。
3. **两个 batch 旋钮都钉死**。原树钉 `global-batch-size` **和** `local-batch-size`;
   我原先只钉了 token 预算,漏了后者 —— 它决定每个 rank 每个微批看多少样本,
   这正是我一度看到 0.42 差异的来源。
4. **种子用 `--checkpoint.create_seed_checkpoint` 造,拷进各格目录**,
   不是 `--checkpoint.initial-load-path`。后者会和格子自己的 checkpoint 目录冲突,
   我因此踩过一次 resume 陷阱。

## 一条记录

我最初断言"老树各格自己初始化、没有统一 ckpt",是**只看了 `run13_flav.sh` 的 BASE**
就下的结论。用户不接受,再查才发现 `verify_refactor.sh` 里就有
`create_seed_checkpoint` + `cp -r` 的完整做法。

**"某个脚本里没有"不等于"这套流程里没有"。**
