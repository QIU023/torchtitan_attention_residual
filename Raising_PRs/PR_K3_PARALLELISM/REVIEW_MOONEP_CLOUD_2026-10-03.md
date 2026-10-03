# MoonEP #4751：云端审查意见的处理（2026-10-03）

用户原话："另一个云端claude审核了PR，有如下意见，你看看是否合理，如果合理，则修复，然后推到moonep pr和review分支"。

结果：两个分支 `k3_moonep_seam`（#4751，仍是 draft）和 `moonep_review1` 都是 `5e4596dc7`。它们仍然是建在 main `db050eb3f` 上的两个提交：

- `95e631a16`：生产代码；
- `5e4596dc7`：测试。

旧 head `94aeed14f` 备份为 `backup/{moonep_review1,k3_moonep_seam}_pre_20261003`。脚本和日志在 `kit_moonep_review_2026-10-03/`。

## 逐条结论

| 意见 | 结论 | 依据 | 改动 |
|---|---|---|---|
| 1. prefetch 只有 trailing barrier，安全性靠一条没写下来的不变量 | 成立 | MoonEP `33327eb` 的 `prefetch.py` 文档和代码：只在结尾有一道 `cross_rank_barrier`。`combine.py` 的 kernel 自带入口 `cross_rank_barrier`，与 `inter_rank_sync` 无关。dispatch 默认 `inter_rank_sync=True`，会在规划前做跨 rank 同步。 | `prefetch_rows` 里，在 `buffer.prefetch_weight(` 上面加一行注释："Prefetch has only a trailing barrier: a dispatch or combine must separate it from the last GEMM over the pools." 按 no-comment 规则，这条是读者不知道就会改坏的约束，只写一行事实。 |
| 2. 没有 MoonEP 版本 pin | 成立 | 2bd860b（08-13）到 33327eb（09-20）之间：`Buffer.__init__` 去掉了 `B`；`prefetch_weight` / `reduce_grad` 从 `full_*` 改成 `local_*` 加 `*_buffer`。两个版本的 `setup.py` 都是 `version="0.0.1"`，也没有新增可以拿来判断版本的公开符号。装旧版的话，第一次前向就会抛 TypeError。 | `moonep.py` 导入时检查 `Buffer.prefetch_weight` 有没有 `local_gate_weight`，没有就报 ImportError，写明需要 33327eb 或更新；安装提示里也写上 33327eb。titan 对 DeepEP 也是这个做法：导入 v2 才有的 `ElasticBuffer`，报错信息写明版本。三种情况已实测：09-20 API 能导入，08-13 风格的 API 报错，没装就提示安装。body 写明了验证过的 commit。 |
| 3. 测试缺跨层、交错和 4 卡 | 成立 | 单层单 micro-batch 时，反向开始时 pool 里本来就是这一层的数据，所以"反向不重填"测不出来。下面的变异检查证实了这一点。 | 新增 2 卡两个用例：两层（权重和路由都不同），以及两层加两个 micro-batch（前向 mb0、前向 mb1、反向 mb0、反向 mb1）。新增 4 卡两个用例：每个 token 的专家分在两个 home rank 上，单层一个、交错一个，这样每个目标 rank 的槽要从两个 home group 接收。 |
| 4. 性能账要写进描述 | 成立 | 代码：`prefetch_rows` 前向调一次，反向重填再调一次，每次都先把本地权重 `copy_` 进 pool；反向重算三个分组 GEMM。不开 AC 时，标准 EP 一共跑 3 次 GEMM（前向，加反向的 dgrad 和 wgrad），这里跑 4 次。 | body 加一段代价说明。参数不 alias pool 的原因写进 Design："FSDP allocates that unsharded storage itself"。没写"不可能"，因为 FSDP2 有没有办法把 all-gather 输出放进外部内存，没有核实。 |
| 小点：`int(plan_id)` 和 compile | 描述补上 | main 只编译局部区域（`LocalCompileConfig` 默认 gated_rmsnorm、loss、swiglu、situglu、cos_sin_rope），K3 的 `parallelize` 总会绑定这些区域，但没有哪个区域包住 MoonEP 的 op。用 `probe_custom_op_minor.py` 实测：一个同样没有 fake 实现、带 ORDERED effect 的 custom op，在 compile 下 `fullgraph=False` 和 `True` 都是 trace 时直接报错（TorchRuntimeError），不是悄悄断图。SiTU-GLU 在 op 内部编译后能跑：10-02 在 H100 上用 `cec583a44` 跑的 trainer 格子已经覆盖到。 | body：No compile region may contain the MoonEP ops, which have no fake implementations; titan's local regions still compile, including SiTU-GLU inside the expert computation. |
| 小点：`grad_route_weights.float()` 假定梯度已被 materialize | 不需要改 | torch 0928 的 `torch/_library/autograd.py`：`register_autograd` 生成的是普通 `autograd.Function`，没有关 `set_materialize_grads`。实测没用到的输出，backward 拿到的是全零张量。 | 无 |
| 小点：pool 是 `[F, D]`，MoonEP 文档是 `[H, H']` | 成立，一句话 | `prefetch_weight` 的文档写 `[epn, H, H']`。gate/up 在 titan 里是 `[e, F, D]`，down 是 `[e, D, F]`，所以只有 gate/up 和文档方向相反。`prefetch.py`："Scale tensors are re-tiled to `[128, nbytes // 128]` per expert"。 | body 一句。 |

## 测试过程中顺带发现并修掉的问题

- **4 卡交错用例 OOM：** 稠密参考按 token gather 专家权重（`[T, F, D]`，T=1024 时每次约 800 MB），在 5060 的 16 GB 上 OOM。现在改成按专家分组计算。float64 下和旧写法差 1.4e-16。
- **`test_moonep_is_refused_outside_kimi_k3` 在有 GPU 的机器上失败**，旧 head `94aeed14f` 也一样：
  - `qwen3_moe_debug` 没有关 CUDA graph。在有 GPU 的机器上，EP=4 加 AllToAll 的原始配置会先被 CUDA graph 校验拦下，走不到 K3-only 检查。
  - CI 的 CPU 机器看不到这个问题。之前记的 "27 passed" 应该是在没有可见 GPU 的环境里跑的。
  - 照旁边 DeepEP 测试的写法加上 `disable_cuda_graphs=True`。

## 检查（全部在 8 × 5060 上用假 MoonEP 包；不进 body）

- GPU 测试（假包，副本只去掉了 multicast 检查）：9 passed（`results/fake_all2.log`），其中 2 卡 7 个、4 卡 2 个。
- 变异检查，测试集为 2 卡 hot、两层、交错，以及 4 卡交错：

  | 变异 | 2 卡 hot（老用例） | 两层 | 交错（2 卡和 4 卡） |
  |---|---|---|---|
  | 反向不重填，直接用 pool 里现有的数据 | 通过 | 失败 | 失败 |
  | 反向用最近一次的 plan，不用自己的 plan | 通过 | 失败 | 失败 |

  失败都是数值比对失败（2.5% 到 5.7% 的元素超出容差），不是崩溃。
- CPU（`test_transforms.py`、`test_moe.py`、`test_integration_test_definitions.py`）：63 passed，9 subtests passed，有无可见 GPU 都一样。其中 `test_transforms.py` 单独跑是 27 passed。
- pyrefly：新旧 head 都是 17 个，完全相同，没有一个在 MoonEP 文件里（`results/pyrefly_*_err.txt`）。
- ufmt、flake8 干净。flake8 的 B905 来自本机较新的 bugbear；仓库 pre-commit 钉的 22.4.25 没有这条规则，main 上同样的写法有 72 处。
- 新增注释只有 `prefetch_rows` 那一行。测试里那行注释是把原注释的 "rank 0" 改成了 "these ranks"。

## 还没做的

- 用真 MoonEP 跑新增的 4 个用例，要一台有 NVSwitch multicast 的 H100，其中 4 卡用例要 4 卡。body 的测试计划写的是 pending。
- body 现在 759 词（原来 646 词），多出来的是审查要求写明的代价、compile 和版本三部分。
