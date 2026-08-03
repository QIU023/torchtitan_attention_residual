# LoRA vs Full-param 后训练:AttnRes Kimi K3 分布式 infra 适配清单

> 2026-07-17 会话分析原样 dump(基于 `parallelize.py` / `pipeline_adapter.py` 实际代码逐轴分析)。

## 先说结论

**Full-param 后训练:分布式 infra 零新增**——它就是预训练 infra + 换 loss(GRPO/SFT 都在最后一个 stage 的 logits 之后,loss-agnostic),唯一新东西在 veRL 边界的权重同步。**LoRA 才是有真实适配清单的那个**,而且工作量集中在两个算子族:MoE grouped experts 和(2.8T 场景的)量化底座。

## LoRA 的逐轴适配清单

### 1. 算子架构层:LoRA 挂哪里 + AttnRes 特例

| 算子 | LoRA 处理 | 适配量 |
|---|---|---|
| MLA `q_proj`/`kv_b_proj`/`o_proj` | 标准 LoRA | 小 |
| MLA `kv_a_proj_with_mqa` | **建议跳过**——输出是 `[kv_lora, qk_rope]` 两半异构 split(我们 TP 也因此 NoParallel),LoRA 的 B 矩阵会跨 split 边界 | 决策即可 |
| KDA 内部 q/k/v/f/g/b projections | 标准 LoRA(都是普通 `nn.Linear`;triton kernel 只作用在投影后的激活上,**不碰 LoRA 路径**)| 小 |
| Dense MLP / shared_experts | 标准 | 小 |
| **GroupedExperts w1/w2/w3** | **最大的实现项**(见 EP 节)| **大** |
| router.gate / embeddings / norms | 惯例排除 | — |
| **`attn_res_proj`(α 伪查询,D→1)** | **不能 LoRA,必须全参训**——D→1 本身秩为 1,LoRA 无意义;且 α 是 graft 的核心可训参数,量极小 | 配置例外项 |

所以 K3 的 LoRA recipe 天然是 **"α 全参 + base LoRA"** 的混合——LoRA 配置系统必须支持 per-module 的 trainable 例外,这是纯配置层的小活,但必须有。

### 2. FSDP2 轴:基本免费

FSDP2 per-param sharding 对 frozen(`requires_grad=False`)参数原生支持,grad reduce-scatter 只发生在可训参数上——**不需要改 `apply_fsdp`**。两个注意点:LoRA A/B 会被自动扫进所在 layer 的 FSDP unit(正确);backward 穿过 frozen 权重仍需要 W 的 all-gather(激活梯度要乘 W^T),**所以 LoRA 省的是梯度+优化器态显存,不省 backward 通信**——这是性能预期管理,不是代码改动。

### 3. TP 轴:小活,但必须动我们手写的 plan

`apply_tp_kimi_k3` 是逐模块手写的 plan dict,要扩展到 LoRA 子模块,配对规则是固定的:

- **Colwise base**(q_proj/kv_b_proj/gate_up):`lora_A` 复制、**`lora_B` colwise 分片**(跟 base 的 out-dim 切法);
- **Rowwise base**(o_proj/down_proj):**`lora_A` rowwise 分片**(吃分片输入)、`lora_B` 复制,LoRA 分支产生 Partial → 要么多付一次 all-reduce,要么把 LoRA 输出加进 base 的 Partial 再一起 reduce(省通信但要动加法位置);
- KDA 整体 NoParallel → KDA 的 LoRA 也复制,零适配;
- 我们的 **plain-boundary 约定**(`use_local_output=True` 让 PP/AttnRes/triton 只见 plain tensor)必须对 LoRA 输出同样成立——plan 写对就行。

### 4. EP 轴:**LoRA 最大的单项工程**

GroupedExperts 是 stacked `[E, d_in, d_out]` 参数走 `grouped_mm`。对它做 LoRA 意味着:

- per-expert 的 A/B 也要 stacked(`[E_local, d, r]` / `[E_local, r, d]`),沿 expert 维在 `ep_mesh` 上和 base 一致分片;
- **低秩分支必须复用 base 的同一次 token dispatch**(路由索引/all-to-all 结果),不能自己再 dispatch 一遍——要 hook 进 `_moe` 的 forward 路径,再补一对 grouped_mm;
- nested-FSDP(`edp_mesh`)的包裹顺序要把 expert-LoRA 参数放进 experts 的 FSDP unit。

**务实建议:第一版 LoRA config 直接跳过 expert LoRA(只 LoRA attention + dense/shared MLP + α 全参)**——这是社区常规做法,把"大项"降级为可选第二版。代价是 MoE 模型大头参数不参与适配,后训练容量受限,A/B 出来后再决定值不值得做 grouped-LoRA。

### 5. PP 轴 + AttnRes adapter:正确性免费,优化是新 feature

这是 HANDOFF §5.4 已论证的,落到代码上:

- **正确性零改动**:LoRA 不豁免跨段 backward——α 的 skip 边梯度是真实 autograd 路径,现有 adapter 的双梯度桥(PP SEND_B + rank-local cache deposit)本来就路由它。冻结 base 不改变激活梯度流。
- **显存语义零改动但占比上升**:retention 由激活决定;LoRA 用户会加大 microbatch → adapter 跨段激活保留在显存里的相对占比变大。写进文档即可。
- **新 feature(可选)**:`first_trainable_stage` 部分反向截断——只有当"早段完全没有可训参数"时才能剪掉早段 backward。**注意和 α 的交互**:α 存在于每一层,所以只有"早段 α 也冻结、只训后段"的 recipe 才触发这个优化。要动 schedule(跳过左侧 stage 的 SEND_B/RECV_B)+ adapter 不给 frozen producer 存 grad 桥。Full-param 永远用不到它。

### 6. 优化器 + checkpoint + veRL 边界

- **优化器**:LoRA = 只对 A/B + α 建 param group(按名字过滤,DTensor 兼容,小活)。顺带:α 是 D→1,将来上 Muon 时要走 per-shape 路由(Muon 只管 2D 矩阵,α 归 Adam)——这是 full/LoRA 共同的小坑。
- **DCP**:LoRA-only 的存取(只存 adapter + α,几十 MB vs 全量 96GB+)+ PP 度重切分要按参数子集过滤——小但必须验(gap 清单里 "DCP 存/取/PP 重切分" 那条的 LoRA 分支)。
- **veRL 权重同步**(glue 但算子相关):full-param = 每迭代全量 reshard 广播(48B 很贵);LoRA = 只同步 A/B + α(KB~MB 级)→ 这就是 HANDOFF 说"**先做 LoRA-only 同步路径**"的原因——但 rollout 引擎要么支持 runtime LoRA 挂载,要么 trainer 侧 merge(W+BA 需要 all-gather 全量 W,又贵回去了)。SGLang/vLLM 的 LoRA serving 路径是现成的,优先走挂载。

### 7. 2.8T 特有:QLoRA 才是真正的硬骨头

48B 的 LoRA 单节点 FSDP 就够;**2.8T 的 LoRA 必须 quantized base(~1.4TB MXFP4)+ PP+EP 摆放**,而 torchtitan 没有 QLoRA 路径——nf4/MXFP4 frozen base 和 DTensor/FSDP2 的组合是**空白**,frozen 量化权重的 dequant-on-the-fly forward + LoRA fp16 分支是独立工程。这是"2.8T LoRA post-training runs on the same stack"承诺背后真正要还的债,7.27 后按需启动。

## 工作量排序(LoRA 相对 full-param 的净增量)

| 优先 | 项 | 量级 |
|---|---|---|
| P0 | α-全参例外的 LoRA 配置系统 + 优化器 param group | 小 |
| P0 | TP plan 扩展(colwise/rowwise LoRA 配对 + plain-boundary)| 小 |
| P0 | LoRA-only DCP 存取 | 小 |
| P1 | veRL LoRA-only 权重同步(rollout 侧挂载)| 中 |
| P2 | grouped-expert LoRA(先跳过)| 大 |
| P2 | `first_trainable_stage` PP 截断(纯优化)| 中 |
| 7.27 后 | QLoRA(量化 base × DTensor × PP+EP)| 大 |
