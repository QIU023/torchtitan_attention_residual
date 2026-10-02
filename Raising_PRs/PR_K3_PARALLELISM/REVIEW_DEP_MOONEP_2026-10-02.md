# DEP（#4381）和 MoonEP（#4751）diff 复审（2026-10-02）

用户："DEP和MoonEP当前的diff已经都是可以提交的吗？本地按照规则再审核一遍，H100只需要跑的是DEP吗"

审的对象：
- MoonEP：`moonep_review1` = `k3_moonep_seam` = #4751 head `e30d82886`，在 main `1aaee42bf` 上。
- DEP：`dep_review1` = `1d03bfcc6`，在 main `97e673b77` 上。PR 分支 `k3_pp_mm` = #4381 head 是 `d27839459`，在 main `46ec3f232` 上，四个提交内容和 `dep_review1` 一样。

上游 main 现在是 `db050eb3f`，比 MoonEP 的基底多 13 个提交，比 DEP 的基底多 33 个。

## 结论

两个都还不能直接转成正式 PR。

**MoonEP：代码没发现缺陷，挡住它的是 CI。**
- 上游 H100 CI（`.github/scripts/run_8xgpu_integration_tests.sh`）的基础那一遍，只排除了 `qwen3_fsdp+deepep` 和 `deepseek_v3_fsdp+hybridep+compile`，排除后再分别装包单独跑。
- 新加的 `kimi_k3_fsdp+moonep` 会落在基础那一遍里。那一遍没装 MoonEP，`init_buffer` 一 import 就报错，CI 必红。
- 另外，这个脚本因为 "CI 机器的 NVLS 不稳定" 设了 `NCCL_NVLS_ENABLE=0`；而 MoonEP 要 NVLink multicast，所以装上包也不一定能跑。

**DEP：有 10 个 pyrefly 错误，都在分支自己的文件里。**
- 上游 lint 用 pyrefly 0.45.1 检查全项目，和本机版本相同，所以这 10 个一定会让 lint 变红。
- 还有几处不合规则的小问题，以及一个没定的数值问题（见下文 DEP 部分）。

**两个都要 rebase。**
- 本地试过，都只有测试格子列表的冲突。
- MoonEP 要注意：新 main 里合进了 #5008，SiTU-GLU 默认编译，而它正好在 MoonEP 的专家算子里跑。所以 rebase 后运行路径变了，要在 GPU 上复查。

**H100 上要跑什么：**
- **DEP 必须跑**，跑的是 K3 区间代价比例那一组，脚本在 `kit_h100_2026-10-02/`。应该在 rebase 并修完 pyrefly 的那个 head 上跑，这样结果属于要提交的那棵树。
- **MoonEP 看是否 rebase：**
  - 不 rebase 就不用再跑。
  - rebase 到含 #5008 的 main 之后，建议做一次短复查：GPU 单测 5 个加 h100 格，十几分钟。先在 5060 上用假包跑一遍。

## 实测的检查

| | MoonEP `e30d82886` | DEP `1d03bfcc6` |
|---|---|---|
| 规模 | 12 个文件，+794 / −8 | 10 个文件，+2020 / −11 |
| pyflakes（改动的文件） | 干净。另有 3 条：两条是 main 原有的 import，一条是带 `noqa` 的 `import moonep` | 干净 |
| pyrefly 0.45.1，对基底比 | 和基底逐条相同（本机环境下两边都是 17 个） | 多 10 个，都在 `dep_plan.py` 和 `vision_dep.py` |
| CPU 测试 | 63 passed，9 subtests（`test_transforms`、`test_integration_test_definitions`、`test_moe`） | 49 passed，28 subtests（`test_kimi_k3_dep_plan`、`test_kimi_k3_vision_dep`、`test_integration_test_definitions`、两个 K3 PP 测试） |
| GPU 测试，不开 GPU 时 | 没有 multicast 时 5 个都干净地 skip | 没跑 |
| 试 rebase 到 `db050eb3f` | 第一个提交（产品代码）没冲突。第二个提交在 3 个文件里冲突，都是格子列表：上游删了 `hsdp+cp+float8` 和 float8 LoRA 两格及它们的 recipe 和 import | 只有 B200 格子列表一处冲突：上游把 `OverrideDefinitions` 改名成了 `IntegrationTestDefinition`。recipe 被 git 自动跟到了 `suites/b200.py` |
| rebase 后（只在本地） | `cec583a44`：pyflakes、ufmt 干净，pyrefly 和 main 相同，CPU 63 passed | `3b61197b9`：pyflakes、ufmt 干净，pyrefly 仍是那 10 个，CPU 51 passed，28 subtests |
| 新增的注释、docstring，logbook 路径、实测数字、TODO | 见下文 | 见下文；两边都没有 logbook 路径、实测数字或 TODO |
| 提交信息 | 没有 trailer，没有跨仓库引用 | 同左 |

试 rebase 的结果留在两个 scratch worktree 里，都是 detached HEAD：`wt_rv_moonep_1002` 和 `wt_rv_dep_1002`。任何分支都没有改，也没有推。

## MoonEP 逐条

**10-02 用户定了：** 第 1 条取办法二，h100 格子和 recipe 删掉，第 8 条因此不用改；另外 `MoonEPRoutedExperts` 搬出 `models/common/moe.py`。都在 `moonep_review1` `7b73b1ed5`，见 `MOONEP_LAYOUT_2026-10-02.md`。

1. **CI（挡住转正式），要用户定，有两种办法：**
   - **办法一：进 CI。**
     - 把这一格加进基础那一遍的 `--exclude`；
     - 新增一个固定 MoonEP 版本（`33327eb`）的安装脚本，仿照 `install_deepep_v2.sh`；
     - 在 DeepEP 那一遍后面再加一遍，单独跑这一格。
     - 风险是 CI 机器的 multicast 不一定可用。
   - **办法二：不进 CI。** 去掉这一格和它的 recipe（规则是：recipe 只给 CI 真会跑的形状），只留 GPU 单测，单测没有 multicast 时会 skip。
2. **rebase：** 冲突都只在列表里，已在本地解开，见上表。
   - #5008 让 `SiTUGLU.__call__` 走 `local_compile("situglu")`，而且默认打开。MoonEP 的 `_compute` 里会调到它：前向在 `no_grad` 下，反向重算在 `enable_grad` 下。
   - 理论上没问题，dynamo 按 grad mode 各编译一份；但这是新路径，H100 的证据没覆盖。
   - 另外 #4961 删掉了 float8 训练，所以"`_grouped_mm` 是 LoRA、float8、mxfp8 覆盖的接缝"这个理由里，float8 已经不存在了。
3. **注释：** 7 条都是一行，写的都是约束。只有一条擦边：`ops.py` 的 "Ops take only tensors, so a plan crosses them as a CPU id into this table; combine removes it."。前半句在讲为什么这样设计，可以只留后半句的不变量。
4. **docstring：** 都是一行（#4577 的标准）。两个类的 docstring 各是一句话折成两行。没问题。
5. **调用别的类的私有方法：** `self.w13._grouped_mm(...)`。规则不许这样做，但它是量化和 LoRA 覆盖的接缝。审核的人可能会问，可以在 body 里用一行说明。
6. **模块级的 plan 表和进程级的 Buffer / 池：**
   - 和 DeepEP 的进程级 buffer 是同一种做法，body 也写了 "as DeepEP's handle does"。
   - 但上游 HybridEP 的 docstring 写的是"handle 走 op、免去全局 handle 缓存"，审核的人可能会对比着问。
   - MoonEP 的 plan 是一个不透明的 Python 对象，走不了 op。这一点可以在 body 里说一行。
7. **配对检查（可选）：**
   - MoonEP 的 dispatcher 配普通的 RoutedExperts，会在 grouped GEMM 处报形状错；反过来配，会报 `AttributeError`。都会响亮地失败，只是报错不直白。
   - `validation.py` 里加一个 isinstance 检查，就能给出明确的报错。
8. **recipe 写法：**
   - 上游把带后端的 recipe 放在 `torchtitan_recipes/tests/models/<model>.py`（`deepseek_v3_debugmodel_hybridep`、`qwen3_moe_deepep`），suite 文件只原地改并行度。
   - MoonEP 是在 suite 文件里直接拼 transform，而且 `replace(...)` 和原地修改混着用。
   - 属于小问题，要对齐的话改成上游那种写法。
9. **K3-only 检查：** 在 core 的 `validation.py` 里延迟 import 了 `KimiK3Model`。这是用户要的，审核的人可能更想要一个模型配置上的标志。保持现状。

## DEP 逐条

1. **pyrefly 的 10 个错误（挡住转正式）。** 都是类型问题，改动小：
   - `dep_plan.py` 里 3 个"breaking cycles"：`_encode_spot` 和 `_backward_spot` 的 `best`，以及一个 `int | None`。给变量加类型标注即可。
   - `vision_dep.py`：
     - `new_subgroups_by_enumeration` 返回的组可能是 `-100` 或 None，要 assert 成 `ProcessGroup`（2 处）；
     - `dist.irecv` 的 `Work | None`（2 处），以及 `_sends.append` 的 `Work | None`（1 处）；
     - `_device` 被当成只读描述符，这是 0928 存根的问题，main 的做法是 `# pyrefly: ignore [read-only]`，或者换个属性名；
     - `replica.merge_kernel_size` 被推成 Module，不可迭代：replica 要标成塔的类型，或者取值时转成 tuple。
2. **rebase：** 一处冲突，本地已解开，见上表。
3. **docstring（#4577 的标准）：**
   - 私有函数 `_slot_times` 和 `_hook_at` 有 docstring，标准是私有函数不写。
   - `DepPlan` 和 `plan_dep` 的 docstring 带了多行段落（`plan_dep` 有 9 行），应该压成一行，再加一小段只写代码看不出的事实。
4. **`__init__.py` 的 `stages,  # pyrefly: ignore[bad-argument-type]`：** 改成用类型写对，比如按 `VisionDepPipelineStage` 筛选或 assert，不要新加 ignore。
5. **自己建的通信组：**
   - `_pp_groups` 假定 pp 是最外层，用 world size 推出所有 PP 组；`install_vision_dep` 再用 `new_subgroups_by_enumeration` 建一个新组。
   - 规则说"不手建通信组"。新建组的理由是有的：DEP 的点对点要和调度自己的 send / recv 用不同的通信器，互不排队；09-27 的审计记过，body 第 20 行也写了。
   - 但这些组应该从 world mesh 取，不要自己推。
6. **包一层 schedule：** `VisionDepSchedule` 用 `__getattr__` 转发，还用了 torch 的私有 API `_batch_p2p`。body 第 22 行写了缺的是哪个接缝（"the engine's pipeline step body has no model hook"），满足规则。
7. **小问题：**
   - `_vision_replica` 调了 `replica._parallelize(...)`，这是 Module 协议自己的方法；
   - 用 `param.data = ...` 转精度；
   - `_tp_dim` 靠形状不一致来猜 TP 切的是哪一维，更好的做法是从并行计划或 placement 推。
8. **核过、没问题的地方：** DEP 在 DP 上对塔的梯度做求和。titan 的 FSDP 设了 `gradient_divide_factor=1.0`，loss 按全局 token 数归一，也是求和，两边一致。
9. **没定的数值问题：**
   - 默认设置下，DEP 开和关第 1 步不一样。原因是塔和文本共用编译好的 `flex_attention`，同时跑两者的 rank 会被 dynamo 重编译。09-30 在 H100 上定位过：关掉 `automatic_dynamic_shapes` 后，第 1、2 步相同。
   - 要定一个做法：给塔单独一份编译的注意力，或者 body 写明这件事，并且做对照时关掉自动动态形状。
10. **body：** 还是 09-28 的 v5。
    - 状态区 10-01 那条"规划器偏离原文"已经过时：10-01 的复审和 5060 实测之后，用户同意规划器不改。
    - Results 要补上 K3 区间的代价比例和 H100 的数字。

## H100 建议

- **DEP：**
  - 先 rebase，再修 pyrefly 和第 3、4 条，这些都不改运行逻辑。然后在新 head 上跑 `calib_base.sh` 和 `run_dep_h100_k3range.sh`。
  - 数值对照要关掉 `automatic_dynamic_shapes`，或者等第 9 条定下来再跑。
  - 塔用 `base` 还是 K3 的，从 10-01 起一直等用户定。
- **MoonEP：** 只有 rebase 之后才需要再跑：GPU 单测 5 个加 h100 格。先在 5060 上用假包跑一遍，用 GPU 前先告诉 SATS-OPRD 会话。
