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
