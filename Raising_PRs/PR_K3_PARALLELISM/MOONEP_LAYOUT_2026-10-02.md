# MoonEP 的文件布局和 h100 格子（2026-10-02，CPU 这边）

用户 10-02 手动审了 MoonEP 的 diff：
> MoonEP我也手动审核了diff ，注意MoonEP只是optional backend，很多user是不一定有这个lib的
> 首先h100 recipe不应该加那个recipe，删了
> 然后torchtitan/models/common/moe.py 里面的改动不应该放这个文件，而且 为什么DeepEP，HybridEP不需要改moe.py 只在 torchtitan/distributed/deepep放改动，而MoonEP需要一个新的TokenDispatcher ？
> _MoonEPExperts 这个类为什么存在？它和 MoonEPRoutedExperts的关系？
> MoonEPRoutedExperts 为什么不放在 torchtitan/distributed/moonep/ 文件夹里面？

## 结果

- review 分支 `moonep_review1`：`cec583a44` → `7b73b1ed5`，快进，加一个提交，等用户测完再并进原来那两个提交。PR 分支 `k3_moonep_seam` 没动，仍是 `e30d82886`。
- 删掉 h100 格子 `kimi_k3_fsdp+moonep` 和 recipe `kimi_k3_moonep_fsdp4_ep4`；`tests/integration_tests/h100.py`、`torchtitan_recipes/tests/suites/h100.py`、`tests/unit_tests/cpu/test_integration_test_definitions.py` 都回到 main。
- `MoonEPRoutedExperts` 从 `models/common/moe.py` 搬到 `distributed/moonep/experts.py`，`moe.py` 回到 main。
- PR 对 main：12 个文件 +790/−7 变成 9 个文件 +769/−6，`models/common/` 下只剩 `token_dispatcher.py` 里的 dispatcher。

## 问答

**DeepEP、HybridEP 为什么不改 `moe.py`？MoonEP 为什么要一个新的 TokenDispatcher？**
- DeepEP 和 HybridEP 也各有一个 TokenDispatcher，就在 `models/common/token_dispatcher.py`（main `db050eb3f`：`DeepEPTokenDispatcher` 第 762 行，`HybridEPTokenDispatcher` 第 877 行）。`distributed/deepep/` 里只放底层的 buffer 和算子，dispatcher 在方法里延迟 import 它们。所以 MoonEP 的 dispatcher 放在同一个文件、继承同一个 `BaseEPTokenDispatcher`，是照它们的先例。
- 它们不改 `moe.py`，是因为只搬 token：标准的 `RoutedExperts.forward` 用本 rank 专家的权重做 grouped GEMM，就能算 dispatch 过来的 token。
- 上游有意这样分：#2842（`[MoE][1/n] Introduce token dispatcher`，2026-04-17）加 token dispatcher 的同时，删掉了 DeepEP 原来专用的 `models/common/moe_deepep.py`。从那以后，后端只加 dispatcher，不加自己的 MoE 子类。
- MoonEP 多出来的是专家这一层，因为它还搬专家权重：
  - 它把热门专家的副本预取到别的 rank 的槽里，每个 rank 的 GEMM 要算"本 rank 的专家 + 槽里的副本"，权重来自 NVLink 池，不是 `self.w13.weight`；
  - 反向时槽的梯度要通过 MoonEP 的 `reduce_grad` 加回本家专家。
- GEMM 由 `RoutedExperts.forward` 管，在 dispatch 和 combine 中间，dispatcher 这个接缝表达不了这一层。所以要一个 `RoutedExperts` 子类，但它没有理由放进所有 MoE 模型都读的 `moe.py`。

**`_MoonEPExperts` 是什么，和 `MoonEPRoutedExperts` 什么关系？**
- 现在的 PR head `e30d82886` 和 review head 里都没有这个类，看到的应该是 09-30 之前的版本。
- 它是 09-28 重写版放在 `moe.py` 里的 `torch.autograd.Function`，一直到 `ab191a771`（09-30）。10-01 换成了 `distributed/moonep/ops.py` 里的 `torch.library` 算子 `moonep::experts`（`15cb3c23c`、`24458aa6e`）。原因是 H100 上 SelectiveAC 按 ATen 算子回放 Python Function 内部的算子，重算时顺序错位（`MOONEP_H100_2026-09-30.md` 第 1、2 节）。
- 它和现在的算子角色一样：
  - `MoonEPRoutedExperts` 是模块，持有 `w13`、`w2`、激活函数和配置，也就是 transform 换进去的那个类；
  - `forward` 是 dispatch → `ops.routed_experts(self._compute, ...)` → combine。
  - 算子是自动求导的边界：前向把权重预取进池，在 `no_grad` 下调回 `MoonEPRoutedExperts._compute` 做 GEMM，不留计算图；反向为自己的 plan 重新填池，开着梯度重算 GEMM，再把槽的梯度加回本家专家。
- 为什么要单独一个边界：池是进程级的，所有 MoE 层共用，下一层的预取会覆盖它。autograd 存下来给 GEMM 反向用的会是池的视图，反向时已经被改写；槽的梯度也要走 MoonEP 的规约，普通 autograd 做不了。

**`MoonEPRoutedExperts` 为什么原来不放在 `distributed/moonep/`？**
- 09-28 重写时照 #4577 的规矩放的：common 模块的子类放在 `models/common`（#4577 的 router 子类在 `models/common/moe.py`）。
- 这个理由对 MoonEP 不成立。#4577 的 router 是纯 torch 的功能，任何模型都能用；MoonEP 是可选后端，要装库、要 NVLink multicast，现在还只限 Kimi K3。上游的可选后端（DeepEP、HybridEP）后端代码都在 `distributed/deepep/`。
- 现在放进 `distributed/moonep/experts.py`，和它驱动的 buffer（`moonep.py`）、算子（`ops.py`）在一起。dispatcher 留在 `token_dispatcher.py`，照 DeepEP、HybridEP 的先例。

## 核对

- 搬过去的类和原来逐行相同，只差 `from torchtitan.distributed.moonep.ops import routed_experts` 从 `forward` 里挪到模块顶部；`ops.py` 顶部只 import torch 和 torch_remat，没装 moonep 也能 import。
- pre-commit 的 hook 逐个跑（trailing-whitespace、end-of-file-fixer、check-ast、insert-license、ufmt、flake8、pydoclint、codespell），全过；pyrefly 0.45.1 对 MoonEP 相关的 6 个源文件只读检查，0 errors。
- CPU 测试（本机，`renderers` 装在 scratchpad 的单独目录，追加在 `sys.path` 末尾，不动全局环境）：`test_transforms.py` 加 `test_moe.py` 是 44 passed、2 failed。失败的是两个 MoonEP 用例，要 import Kimi K3（本机缺 `attn_gym.linear._delta_rule.gate` 和 torchvision）；改动前的 `cec583a44` 在本机也是同样的 44 passed、2 failed。`test_integration_test_definitions.py` 本机缺 torchvision 收集不了，它和两个 h100 文件现在都和 main 逐字节相同。
- 用 DeepSeek V3 debug 配置直接跑 transform（`kit_moonep_layout_2026-10-02/moonep_transform_check.py`，给 `KimiK3Model` 打了桩）：新旧两棵树都报 "enabled for Kimi K3 only"，5 个 routed experts 都转成 MoonEP 的专家和 dispatcher，每 rank 16384 个 token；建出来的类分别是 `torchtitan.distributed.moonep.experts.MoonEPRoutedExperts` 和原来的 `torchtitan.models.common.moe.MoonEPRoutedExperts`；过程中都没有 import moonep 库。
- 待 GPU 机跑：`test_transforms.py` 全量（27 个），GPU 测试 5 个（只改了 import 路径）。
- 顺带看到的一点（没改）：validation 里"只限 K3"的检查延迟 import `KimiK3Model`。在装不了 K3 依赖的机器上给别的模型开 MoonEP，报的会是 attn_gym 的 import 错，不是那句 ValueError。只有用 MoonEP 的配置才走到这里。
