# MoonEP dispatcher draft:接口与待验项

从 `kimi_k3/moon_ep_dispatcher.py` 的模块 docstring 搬出。原文 48 行。内容未改。

Our EP is torchtitan's ``AllToAllTokenDispatcher``: reorder, dispatch all-to-all, combine
back. That is correct and it is not what the report describes. Sec 5.2.1 pairs EP at 896
experts with a balanced dispatch, and sec 2.3 says the sparsity K3 runs (896 experts,
top-16) is "beyond the range where the existing auxiliary-loss-free bias update still works
well" -- so the gap is not only throughput, it touches whether the router stays balanced at
all. ``quantile_balance.py`` addresses the router half (it solves for the bias instead of
nudging it); this addresses the transport half.

MoonshotAI released the transport as MoonEP (https://github.com/MoonshotAI/MoonEP).

## Why it is a dispatcher and nothing else

torchtitan already has the seam: ``BaseEPTokenDispatcher`` is an ABC with exactly
``dispatch`` and ``combine``, ``wire_meshes`` installs the EP mesh, and ``init_buffer`` is
the hook a persistent-buffer backend needs. DeepEP is already wired through that seam for
inference (``torchtitan/overrides/moe_token_dispatcher.py``), and ``deep_ep.*`` is already in
``pyproject.toml``'s optional-import list. So MoonEP does not need a new abstraction, a new
dependency mechanism, or any change to the MoE module -- it needs one subclass and one line
in that list.

That is the whole reason this file is small. An earlier instinct was to write an EP path;
the work was to find the seam that already existed.

## Status: DRAFT, never executed

MoonEP needs 8 NVLink-connected GPUs and this box has none of that topology, so nothing
below has run. What IS checked: the interface matches ``BaseEPTokenDispatcher``'s signatures
as of this tree, and the import is optional in the same way fla's and DeepEP's are, so
importing this module on a machine without MoonEP does not break collection.

What a reviewer should NOT read into it: any claim about balance or throughput. The point of
writing it now is that the integration shape is decidable without the hardware, while the
numbers are not.

## The two things to verify first when hardware exists

1. **Token conservation.** ``dispatch`` then ``combine`` must return one row per input row,
   in input order, for every routing pattern including an expert receiving zero tokens.
   ``AllToAllTokenDispatcher`` is the reference: run both on the same inputs and compare.
2. **The backward.** ``combine``'s gradient has to reach ``dispatch``'s input. MoonEP's
   kernels are what own that; if they do not carry a backward, this needs an
   ``autograd.Function`` wrapper and the repo has already paid for getting that wrong once --
   a hand-rolled ``dist.all_gather`` halo in KCP dropped the gradient owed to the left
   neighbour while the forward stayed bit-exact.


---

# 2026-08-28 更新:DRAFT 已在新树写成实现

上文的 DRAFT 状态到此为止。实现落在**新树**(`QIU023/torchtitan` `main`/`k3_on_4025`
@ `d51e27b35`),文件 `torchtitan/models/kimi_k3/moon_ep_dispatcher.py` + 注册表
接线 + CPU 测试 3 个。老树的 137 行骨架不再是权威版本。

## 实现方式

- 按 **MoonshotAI/MoonEP 已发布 README 的真实 API** 写,不再是猜测的签名:
  - `Buffer(S, H, K, E, num_ep_ranks, num_sms=32, token_padding=128)` 一次性
    预分配;`S` 即 base Config 的 `num_max_tokens_per_rank`(存储上界,非每步
    token 数)。
  - `buffer.dispatch(hidden_sh, route_weights_sk, topk_experts_sk,
    tokens_per_expert)` -> `(hidden_nvsh, route_weights_nvs, cu_seqlens, plan)`;
    dtype 约定 bf16 / fp32 / int32 / int32。
  - `buffer.combine(plan, hidden_nvsh, route_weights_nvs)` -> `(output_sh, …)`,
    路由权重在 combine 侧乘——与 standard `AllToAllTokenDispatcher` 的
    combine 侧 fp32 乘 + scatter_add 同构,所以合同对得上。
  - README 明示内核带反向("dispatch bwd"/"combine bwd"),故**不叠**
    autograd.Function;反向仍是上机验证项 #2,不因文档而免验。
- 新树合同的一个关键差异(老 DRAFT 没有):`dispatch` 第二返回值是
  **每本地专家 token 计数**(供 grouped GEMM),不是 routed_scores。由
  `cu_seqlens` 差分得到,长度断言埋作 tripwire。
- 位置:模型文件夹(同 fla 先例),core 的 `make_token_dispatcher_config`
  零改动;spec 选择 `model_registry("debugmodel", moe_comm_backend="moonep")`。
- **尺寸修正**:K3 的 routed experts 消费 `routed_down` 之后的 **latent 流**,
  buffer 按 `latent_dim` 配尺,不是 model dim(moe.py:136 实读确认)。
- `prefetch_weight` / `reduce_grad`(专家权重预取半场)刻意未接:它改变专家
  权重的持有 rank,归下一单元,先立 dispatch/combine 对价。
- EP=1 走 local 回退,不 import moon_ep——CPU 上已实测 round-trip
  (token 守恒的 EP=1 面已覆盖)。

## 上机调试清单(代码内逐行标注 `ON-BOX`)

1. **Buffer 的进程组绑定**:README 签名只有 `num_ep_ranks`、无 group 参数——
   确认绑默认组还是有未写出的 kwarg;EP 组在 `self.ep_mesh.get_group()`。
2. **hidden dtype**:README 只示 bf16,确认其它 dtype 是否被拒。
3. **`cu_seqlens` 布局**:按"本地专家边界(长度 = 本地专家数+1)"读,
   tripwire 已埋,不符会报错并指明改哪一行。
4. 然后按序两项硬验证(顺序不可换):
   a. **token 守恒**——同输入对照 AllToAllTokenDispatcher,含零 token 专家格;
   b. **反向到达 dispatch 输入**——KCP halo 丢梯度的前车之鉴。

## 租机日程(2×H100 SXM,NVLink 经 NVSwitch)

验收:`nvidia-smi topo -m` 两卡间须 `NV#`。装 DeepEP v2(≥2.0.0)+ MoonEP。
顺序:deepep / minimal_async_ep / standard 各 ep2×10 步收格(后端验证 ep2
足够,正文表**不**逐后端重测,见 EP PR comment 约定)→ MoonEP 按上表调试 →
moonep ep2 对照 standard 数值格。hybridep 不在此机(GB200/NVL72 专属,委托
Elfie)。PCIe 本机的 minimal_async_ep 结果已有:死于
`init_buffer -> symm_mem.rendezvous, CUDA driver error: invalid device
ordinal`(无 P2P,非 bug)。

---

# 2026-08-28 审核(Windows 侧,对照 `MoonshotAI/MoonEP` master 源码):`d51e27b3` 的合同错位与修正 `b4e104a6`

依据不再只是 README,而是 `moonep/api.py`(`Buffer.__init__` 在 :440 起)、
`moonep/buffer.py`、`tests/test_e2e.py`。

## 设计层(未解决,已做成显式报错)

MoonEP 的均衡来自**把热点专家动态复制到别的 rank**:训练强制 `B = E/R` 个预取槽,
每个专家投影要求"one contiguous symmetric-memory `[E+B, H, H']` tensor, identically
laid out on every rank",group GEMM 按行号寻址,`prefetch_weight` 在 GEMM 前、
`reduce_grad` 在反向。`RoutedExperts` 的 `w1/w2/w3` 是 EP+FSDP 切片的 DTensor,
GEMM 只跑本地专家——**"只要一个 dispatcher 子类、MoE 模块不动"的前提不成立**,
也不存在"不接 prefetch 的纯均衡 a2a 模式"(B 默认即 E/R)。现在 `init_buffer` 在
EP>1 时 `NotImplementedError` 并说明缺的是哪一层;`allocate_buffer` +
dispatch/combine 保持可直接调用,供独立对价实验。

## 接口层(已修)

| 项 | 原实现 | 源码事实 | 修正 |
|---|---|---|---|
| 包名 | `import moon_ep` | `from moonep import Buffer` | `moonep`;测试同步 |
| 进程组 | "README 无 group,ON-BOX" | `Buffer(..., B=None, group=None)`,None 走默认组;`num_ep_ranks` 须等于 group 大小 | 传 `group=ep_mesh.get_group()` |
| S | 上界,默认 8192,`<=` 检查 | 静态形状,输入恒为 `[S,H]` | 由 `update_from_config` 从 `num_tokens_per_microbatch_per_dp_rank // (cp*tp)` 填;`!=` 即报错 |
| dtype | "ON-BOX 是否只收 bf16" | `assert hidden_nvsh.dtype == torch.bfloat16` | 进出 cast |
| `cu_seqlens` | 按本地专家数+1 读,tripwire | `[E+B]`,每 VM 组行的 padded end offset | 按 E+B 差分,返回每行 token 数 |
| 反向 | "内核自带,不叠 wrapper" | README 的 dispatch bwd / combine bwd 是给框架的**配方**;api.py 无 autograd.Function | 两个 `autograd.Function`:dispatch bwd = `combine(plan, grad)`,combine bwd = `dispatch(grad, plan=plan)` |
| 路由权重 | 交给 MoonEP combine 内乘 | combine 的 `route_weights_nvs` 可选;反向配方不含权重梯度 | 在 torchtitan 侧 autograd 相乘(与 standard 同构),router 梯度路径不变;权重梯度经 `combine(..., route_weights_nvs=grad)` 的 gather 回 `[S,K]` |
| plan 传递 | 猜 `buffer.last_plan` | 无此属性 | Function 通过调用方传入的 list 带出 |

## 2×H100 能做的事(改变目标)

按现状 `moe_comm_backend="moonep"` 在 EP>1 会按设计报错。租机目标改为:
1. 跑 MoonEP 自带 `tests/test_dispatch.py` / `test_combine.py` / `test_e2e.py`
   (写死 8 rank,先看能否 R=2);
2. torchtitan 之外的独立对价:两卡专家权重整表复制成 `[E+B,H,H']`,
   `allocate_buffer` → dispatch → 按 `cu_seqlens` 索引的手写 grouped GEMM → combine,
   对照 `AllToAllTokenDispatcher` + 本地 GEMM 验 token 守恒与数值,再验
   autograd 配方的梯度到达。
这两步立住之后,才知道 torchtitan 侧要动的是专家权重存储与 FSDP 的关系,
那是 MoonEP 后端真正的单元。

---

# 2026-08-28 实现:专家侧单元落地(`main`/`k3_on_4025` @ `0c30608a`)

上一节的"设计层未解决"到此关闭:`moe_comm_backend="moonep"` 现在选中
**dispatcher + 专家模块**两件,EP>1 不再报 NotImplementedError,而是走完整路径。

## 结构

| 文件 | 内容 |
|---|---|
| `kimi_k3/moon_ep_experts.py`(新) | `MoonEPGroupedExperts`:每个投影一张 bf16 `[E+B, in, out]` 计算表(本地行每步从 fp32 master 参数刷新,与 FSDP 混精同构);`prefetch_weight` → 三个 grouped GEMM(`offs = cu_seqlens`)→ 反向重算(同 AC)得 `[E+B]` 表梯度 → 本地行给参数、槽行进 reduce buffer → `reduce_grad` 拉回 home rank。`check_moonep_mesh`:第一版只支持 `dp_shard == ep`(efsdp=1)且无 dp_replicate,其余 parallelize 时拒绝。`MoonEPTableBackend` 是分配层接口。 |
| `kimi_k3/moon_ep_dispatcher.py` | `init_buffer` 经 `_buffer_factory` 分配(测试可替换);`current_plan()` 把在飞的 plan/cu_seqlens 交给专家侧。 |
| `kimi_k3/moe.py` | `KimiLatentMoE.parallelize`:子模块并行化之后把专家模块 attach 到 dispatcher(函数内 import 是因为 moon_ep_experts 反向依赖本模块)。 |
| `kimi_k3/__init__.py` | `"moonep"` 同时选 `MoonEPGroupedExperts` 作 inner_experts。 |
| `kimi_k3/tests/moonep_fake.py`(新) | `Buffer` 与分配层的进程内替身:R 个 rank = R 个线程,集合通信 = barrier(按调用代次键控),复制哪些专家由测试的 `dup_map` 指定,被复制专家的 token 在 home 与副本间交替;行内 padding 也模拟。 |
| `kimi_k3/tests/test_moon_ep_dispatcher.py` | 新增端到端:两 rank 走 dispatch → prefetch → experts → combine → backward,对照全部专家 fp32 权重的逐 token 稠密参考,断言输出、输入梯度、**专家梯度(含别的 rank 在槽里替它算的那部分)**都对上。 |

## 先在 PCIe 盒子上跑(不需要 NVLink,不需要 moonep 包)

    pytest torchtitan/models/kimi_k3/tests/test_moon_ep_dispatcher.py -x -q

Windows 侧只有 torch 1.9,这组 CPU 测试**没有被执行过**;它们是租机前必须过的门。
盲写的接线错误(dtype、形状、autograd 返回个数、线程死锁)都会在这里而不是 H100 上暴露。

## 租机(2×H100)剩下的唯一分配题

`MoonEPTableBackendNVLink.alloc_*` 目前 raise。MoonEP 合同要求每个投影是**一段**连续
虚拟地址的 `E+B` 行:前 E 行按 rank 分块映射各自的物理内存(`prefetch_weight` 直接
`full_weight[:E]` 切片读远端),后 B 行是本地槽页。`moonep.buffer.create_nvl_dist_tensor`
只映射等长分块、没有槽尾;MoonEP 自己的 e2e 用本地 `torch.empty` 假造了这段(它只测通信)。
两条路二选一,上机定:
1. `create_nvl_dist_tensor` 分块取 `E/R(padded) + B` 行,并调整 plan 的行号映射;
2. 用其 VMM 原语自己 reserve 一段,把 R 个分块和槽页 back-to-back 映射。
之后的验证顺序不变:token 守恒 → combine 梯度到达 dispatch 输入 → 与 `standard` 的
ep2 数值格对照。
