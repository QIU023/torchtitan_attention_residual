# 栈分支 `k3_mxqat_stack_20261012` 重新评估（2026-10-19，CPU 会话）

用户："重新评估栈分支，对照logbook栈分支相关的commits，因为DEP和Dynamic CP经历了rebase大改，我现在不清楚这个能不能用"。

栈头 `2eff57f99`（10-18 推送），base 是 upstream main `b663c1673`（10-12）。这次只读代码和记录，没有跑 GPU。

## 结论

- **能用**，前提是用它自己的 base 和它自己的固定依赖（第 5 节）。
- DEP 和 dynamic CP 这几轮 rebase，改的是 PR 的 base 和新 main 的 API 名字，机制代码没有变。内容上的修改栈上都已经有：
  - dynamic CP 和 #5266 冲突的修复（方案 A）：10-16 折进栈，10-18 推送；
  - DEP 和 dynamic CP 一起开时的两处（副本拿 sub-CP groups；开 dynamic CP 时副本梯度在 dp × cp 上归约）：只在栈上有，两个 PR 都不带，因为只有两个 PR 都在时才有意义。
- 栈不用重新 squash。
- **能引用的验证：** dynamic CP 关的全部；dynamic CP 开的，只有 10-16 直接在 `2eff57f99` 上跑的那几项（第 3 节）。
- **不能再引用的：** 10-15 的 3584² 图片和 1792² 视频那组（跑在修复之前的 `c7ca6fb1a` 上）；CI 两个 CP 格子在修复之前和 main 对比的数。

## 1. 栈的历史（对照 logbook 提交）

| 日期 | 栈头 | 变化 | logbook |
|---|---|---|---|
| 10-12 | `92afeeeee` | Elfie 的 9 个提交 + 适配 + PR-4380 / 4381 / 4751 各一个 squash | `7b206b2ce`（`MXQAT_STACK_2026-10-12.md`） |
| 10-12 | `66f0a47a4` 到 `0fa53c2e3` | 叠 PR-4656 / 4963，再叠 LoRA | `1c4c6a622`、`6ec7d05e8`、`81e7a94dc`、`40fa344c8` |
| 10-13 | `b49021d5e` | LoRA 压成一个提交 | `0d1b98236` |
| 10-14 | `0d00a4f09` | 拿掉 MoonEP（PR-4751） | `bc3e3e003` |
| 10-14 | `7f8b1bf4d` | Elfie 脚本的移植和 fused loss 修复折进适配提交；全部重跑 | `980a3eb1e`、`089587f04`（`VERIFY_STACK_2026-10-13.md`） |
| 10-14 | `a89d39417` | PR-4963 冻结前缀的修复 | `1da1f195b` |
| 10-15 | `c7ca6fb1a` | DEP 副本拿 sub-CP groups | `6fe74f5c1`、`4da1dfe82`（`DEP_DYNCP_COMBO_2026-10-15.md`） |
| 10-15 | 同上 | 3584² 图片、1792² 视频格子；发现和 #5266 冲突 | `4cf90f184`、`03a0be1b5`、`05a060eee` |
| 10-16 本地，10-18 推送 | `2eff57f99` | #5266 冲突的修复；DEP 副本在 dp × cp 上归约 | `895a53740`、`0f249d8f2`（`DYNCP_5266_FIX_2026-10-16.md` 第 5 节） |

## 2. 逐提交和 PR 现在的 head 对照（10-19）

- 方法：栈提交的增删行，按文件和 PR 的 `git diff <base> <head>` 比较；同一边先加后删的行互相抵消，只是挪了位置的代码不算差别。脚本 `kit_stack_reassess_2026-10-19/cmp_stack.py`。
- PR 头都是 10-19 从 GitHub API 读的。

| 栈提交 | 内容 | 对照 | 结果 |
|---|---|---|---|
| `de631c30f` + `0874e6727` | PR-4380 dynamic CP，含 #5266 修复 | `k3_cp_mm` = `cpmm_review1` = `a2a97b88a`（PR 头，ready），base `e61df55b6` | 11 个文件里 9 个逐行相同。GPU 测试一个参数名：栈 `ac_config=None`，PR `activation_remat=None`（main 的 #5102 在栈 base 之后改名）。`vision_encoder.py` 栈上多一个空行 |
| `9bea80240` | PR-4381 DEP | `k3_pp_mm` = `dep_review1` = `7529daeea`（PR 头，ready），base `efdde5eda` | plan、schedule、stage、`model.py`、GPU 测试、B200 那一行逐行相同。其余差别见下 |
| `6dbf362bb` + `c018b28ca` | PR-4656 列表载体 | `attnres_review1` = `507a2f7d5` | 5 个文件逐行相同。线上 PR 分支 `k3_ac_reuse_attention` 仍是 `7dea4cbaf`（#5033 之前的版本，等 shuhuayu），栈带的是 review 头 |
| `65280d259` | PR-4963 | `pp_review_optimize` = `62b32e5af`（PR 头，draft） | 7 个文件逐行相同，含冻结前缀的修复 |
| `2eff57f99` | PR-4576 LoRA | `lora_review2` = `676e443e0`（PR 头，ready） | 10 个文件逐行相同。栈多一个测试 `tests/unit_tests/cpu/test_mx_qat_transform.py`（+26，合并把 adapter 折进 MX QAT 的主权重） |
| `ae2a0002c` 到 `d3b5271d6`，`70e7263cd` | Elfie 的 MX QAT 9 个提交，加适配 | elfiegg/torchtitan `work/kimi-k3-mx-qat-distmuon` | 源头没动：`ls-remote` 仍是 10-12 拷的 `b7b5eb2b9`。她的 draft PR #4791 头仍是 `d7d53e823`（09-29），10-18 的更新只是 bot 改了评论。她的 torchao `pr/mx-fake-quant-ste` 仍是 `6352062` |

- DEP 的其余差别，只有三类：
  - 旧 API：栈上副本收 `ac_config`、`dump_folder`（`ac_config.build(dump_folder=...).apply(replica)`），PR 收 `activation_remat`（`activation_remat.apply(replica)`）；CPU 测试同样。
  - PR 两处 `zip(..., strict=)`：新 main 的 flake8 选了 B905。
  - 栈独有：`pipeline_kimi_k3` 给副本调 `install_vision_cp`；设了 `dynamic_cp_min_patches` 时，副本梯度用 `loss` mesh（dp × cp）归约，否则用 `dp`。
- PR 里没有栈缺的代码：除了上面的 API 名字，只在 PR 一侧的行是 0。

## 3. 两轮 rebase 为什么不用跟

- PR-4380：
  - 10-16 rebase 到 `95788b053`，加了修复提交 `f4010b98c`；同一修复当晚折进栈（`de631c30f`）。
  - 10-18 rebase 到 `e61df55b6`，没有内容修改（head `a2a97b88a`）。
- PR-4381：10-19 rebase 到 `efdde5eda`（`REBASE_PLAN_4380_4381_2026-10-18.md` 最后一节）。
  - 冲突照栈上已经适配过的写法解：`model.py` 两边都留，顺序和栈相同；`BaseModel` 用新路径；B200 那一行挪到 `tests/integration_tests/b200.py`。
  - 适配提交 `7529daeea`：`activation_remat`、`BaseModel` 路径、`zip(strict=)`。都是新 main 的接口，栈的 base 上用不到。
- 所以栈和两个 PR 现在的差别，全是 base 不同带来的。只有栈本身 rebase 到新 main 时才要跟（第 6 节）。

## 4. 验证在当前栈头 `2eff57f99` 上的范围

### 直接在 `2eff57f99` 上跑的（10-16）

- CPU 116 passed（90 subtests：K3、vision_cp 规划、DEP、DEP 规划）；GPU 5 passed（vision_cp 3 个，含新加的梯度测试；DEP 2 个）。
- 两个开类型检查的 CP 格子（all-gather KV、Ulysses）：10 步，没有类型错误，切图路径每个 rank 走了 5 次。
- 8 格：CP2（dp2 ep2 pp2 cp2）和 CP4（ep4 pp2 cp4），DEP 开关 × dynamic CP 开关：
  - 全部跑满；第 10 步后，持有 stage 0 的 4 个 rank 上 tower 参数 22/22 逐位相同；
  - 开 dynamic CP 时，DEP 在 dp × cp 上的求和是对的。判断靠量级：漏了或多加一次 CP 求和，相对差会到 0.5 以上，实测和底同级（1e-3 量级）。

### 按代码沿用（跑在 `7f8b1bf4d` 和 `a89d39417` 上）

- 依据：`git diff a89d39417 2eff57f99` 只动 5 个文件（两处修复和新 GPU 测试），每一处都只在设了 `dynamic_cp_min_patches` 时才执行：
  - `_encoder_is_cp_invariant` 在关的时候返回 True，FSDP 的 mesh 和以前相同；
  - `MoonViTCPEncoder._parallelize` 的类型改写，只在开 CP 且设了阈值时执行；
  - `install_vision_cp` 在关的时候直接返回；副本的归约 mesh 仍是 `dp`。
- flavor 不设这个值。栈上只有 h100 的两个 CP recipe 设 256（`tests/integration_tests/h100.py:73`、`:103`）；18 格的入口（`kit_lora_mm18_2026-10-12/mm18_probe.py`）也不设。
- 所以 `VERIFY_STACK_2026-10-13.md` 里这些照旧成立：
  - 2.8T key 映射；CPU 全量（只在栈上失败的 0 个）；GPU 单测的两条 CI 线（只在栈上失败的 0 个）；
  - 18 格的 base、MX QAT、MX QAT + DistMuon、LoRA、QLoRA；冻结 stage 两格（PR-4963 的修复之后）；续跑（差别已定位到编译的自动动态形状）；
  - CI 格子：mm、mm_muon 20 步和 main 逐位相同；pp × vpp4（开 DEP）第 1 步相同；Elfie 的 MX QAT recipe；MXFP4 checkpoint 集成测试；
  - Elfie 的入口：preflight、dp1、full128、full256（PP2 × EP4、PP8）、memory32 冒烟。这些都是纯文本，ViT 不跑。

### 不能再引用

- 开了 dynamic CP、又跑在修复之前的：
  - CI 两个 CP 格子 10-12 和 10-14 和 main 对比的数（第 5 步起不同的那几行）。修复后只跑过类型检查的 10 步，没有重新配对。
  - 10-15 的 DEP + dynamic CP 组合表（`DEP_DYNCP_COMBO_2026-10-15.md`），已被 10-16 的 8 格取代。
  - 10-15 的 3584² 图片和 1792² 视频（`BIG_INPUT_3584_2026-10-15.md`，树 `c7ca6fb1a`）：
    - 开 dynamic CP 的格子当时梯度是错的（切开的部分放大 CP 倍，各 CP rank 的 tower 分叉）；
    - 修复后，开 dynamic CP 时 tower 的梯度改在 dp × cp 上归约（不开 DEP 时 FSDP 在 dp_shard × cp 上切，开 DEP 时副本用 `loss` mesh），显存和每步时间都会变；
    - 所以表里的显存和时间不能再给 Elfie。要用就在 `2eff57f99` 上用同一个 kit 重跑。
- 视频：栈上没有视频 PR。
  - 10-15 那组是在临时 worktree 里叠了一个没推过的旧视频提交（`cae4f91d1`）；
  - 现在的视频 PR（`k3_video` = `e117882e6`）改了抽帧和时间戳。栈上要跑视频，得先把它叠上去。

## 5. 没定位的（和 PR 共有，不是栈带来的）

- DEP 开和关之间：
  - dynamic CP 关时，tower 梯度相对差约 2e-3（CP2 22 个里 1 个逐位相同，CP4 7 个）；
  - loss 从第 2 步起差 1e-4 量级（10-15 在栈上、10-19 在 rebase 后的 PR-4381 上都是这样）；
  - PR-4381 body 里，10-04 在 H100 上 DEP 开关 100 步逐位相同，和本机对不上。来源没有定位。
- 第 4 节对 DEP 求和的判断只靠量级，不依赖这个底从哪来。

## 6. base 落后 main 81 个提交（`b663c1673` 对 `619cd4cef`）

- **环境要用栈自己的固定依赖：** spmd_types `2ae49fce3`、torch_remat `4fd000acb`、attn-gym 0.0.16（main 现在是 `fce1acb1e`、`624f19cbd`、0.0.17），另加 Elfie 的 torchao 源码。验证时的 torch 是 2.15.0.dev20261005。
- **栈上缺、和 K3 大规模训练有关的 main 修复：**
  - #5393：NCCL 2.32.3 之前，普通 Ulysses 可能算错或者非法访存。main 在建进程组之前设 `NCCL_GRAPH_REGISTER=0`，栈上跑 Ulysses 要自己设。
  - #5350：第一次 optimizer 更新前释放 CUDA 缓存，让 NCCL 在梯度裁剪时能分配缓冲。显存紧的时候可能有关（memory32 峰值 174 GB，每卡 186 GB）。
  - #5358：deterministic 模式下 KDA L2Norm 反向不再 autotune（要 attn-gym 0.0.17），只影响逐位复现。
  - 其余（#5279 DistMuon 用 bf16 传输、#5399、#5398、#5408）是性能，不影响能不能用。
- **rebase 的代价：** `git merge-tree` 有 6 个文件内容冲突：
  - `kimi_k3/model.py`、`kimi_k3/state_dict_adapter.py`、`torchtitan_recipes/tests/models/kimi_k3.py`、`observability/profiler.py`、`tests/unit_tests/cpu/models/common/test_lora.py`、`tests/unit_tests/cpu/models/test_state_dict_adapter.py`；
  - main 那边的来源：#5445、#5345、#5335、#5102、#5307、#5063（to_hf / from_hf 的量化参数，和 Elfie 的 MXFP4 导入重叠）、#5279、#5072、#5444；
  - 大部分落在 Elfie 的提交上。DEP 和 dynamic CP 直接换成 PR 现在的 head（已经适配新 main），组合那几行照搬。
- 建议：不 rebase，除非 Elfie 要用新 main 的东西。

## 7. 以前记下、现在仍然有效的缺口

- MX QAT 开不了 spmd 类型检查：要在 `quantization/mx_qat` 注册她 torchao 的三个 autograd Function（10-13 探针验证过）。
- LoRA + TP + 类型检查：main 的 LoRA 原有的缺口。
- 移植后的 recompute-matmuls 和她原来的按算子策略不完全等价（attention 里的 bmm 不再重算），要她确认。
- 她的脚本如果进 PR，CI 的 pyrefly 会报 43 个错；`KimiK3StateDictAdapter.get_hf_storage_reader` 少 `thread_count` 参数。
- 要逐位续跑，得从第一次编译就开动态形状，或者关掉编译。
- 高分辨率数据要调小 loader 的 `read_options`、`streaming_shuffle_buffer_size`、`num_prefetch_microbatches`，否则主机内存不够。
- MoonEP 10-14 已从栈上拿掉。

## 8. 要用户定的

- 要不要在 `2eff57f99` 上重跑 3584² 图片那组（本机 8 卡，kit 现成）：只有要把高分辨率的显存和时间交给 Elfie 时才需要。
- 栈要不要 rebase 到新 main：默认不做。
