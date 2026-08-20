# CP 的声明式改写(2026-08-20)

配套代码:`torchtitan/models/kimi_k3/sharding.py`,`parallelize.py::apply_cp_kimi_k3`,
测试 `tests/test_cp_contracts.py`。

## 为什么现在做

上游已经把旧路拆了,不是将来要拆。三个提交连在一起:

| 提交 | 后果 |
|---|---|
| `5dd944e62` PR-4218 | 删除 `apply_cp_to_forward`,即 partial_dtensor 的 CP 路径 |
| `601cf4d23` PR-4217 | 删除 `full_dtensor` 后端 |
| 新增 `validate_cp_backend` | CP 现在强制 `spmd_backend='spmd_types'` |

我们的 kimi_k3 从没调用过 `apply_cp_to_forward`(CP 全在模块内),所以 PR-4218 不会当场
把我们打断。**但这正是问题**:我们的 CP 绕开了整个 placement 体系,新后端看不见它。

而 gate 的 21 个 CP 格子全部 pin 在 `--parallelism.spmd_backend partial_dtensor` 上,
按上游新规则那个组合下 CP 直接 `ValueError`。

## 改了什么,没改什么

**没改**:`_cp_all_to_all_headseq`、`KimiMLAAttention._forward_cp`、
`KimiDeltaAttention._forward_cp`、`kcp.py` —— 一行未动。集合通信仍由注意力模块自己发出。

**改了**:CP 的*声明*从散在各处的命令式分支收成一处契约。

`sharding.py` 把每种 CP 算法写成 CP 轴上的一个 placement 对:

    ULYSSES   in: S(1) -> S(2)    out: S(2) -> S(1)    head_sharded=True
    KCP       in: S(1) -> S(1)    out: S(1) -> S(1)    head_sharded=False

Ulysses 恰好*就是*一次 DTensor 重分布:序列维(1)换成头维(2),这在 DTensor 里下降为
`all_to_all_single` —— 与我们手写的那个是同一个集合通信。所以它能被声明式表达。

KCP 不能。delta-rule 递推把状态逐 rank 传递,那是**顺序依赖,不是重分布**,没有任何
placement 对描述得了它。所以它被声明为恒等对,扫描留在 op 内部。声明它只是为了两种模式
有同一个形状 —— 而不是假装它是一次重分布。

前置检查因此从"按算法手写分支"变成查 `head_sharded`:KCP 从不切头,在它上面强制头整除
会拒掉本来能跑的配置。这条以前是靠一句注释维持的,现在是数据。

`model.py` 里那份重复的 `cp_mode not in ("ulysses","kcp")` 校验也接到了
`contract_for_mode`,可接受模式只在一处声明。

## 这个声明目前不是承重的

必须说清楚:契约现在被**校验**、被**记入日志**,但**不驱动任何数值行为**。
`_forward_cp` 里仍然硬写着 `seq_to_head=True`;把 `ULYSSES` 的 pair 改成 `S(1)->S(3)`,
运行时不会有任何变化。

让它承重的办法是让 `_forward_cp` 从契约读方向,而不是读字面量 —— 那不改算法,但要动
`_forward_cp` 本身,超出"Ulysses/KCP 逻辑不变"的约束,所以没做。

在那之前,把两者绑在一起的是 `test_cp_contracts.py`:Ulysses 那条断言的是
`_cp_all_to_all_headseq` **真实的 reshape 结果**,不是断言声明等于自己。声明漂了,测试会红。
这是约束下能拿到的最强保证,但它是测试级的,不是运行时的 —— 不要把它说成"声明驱动实现"。

## 只走到一半,以及为什么

完整的声明式形态是让边界(`local_map`)发出集合通信,模块内不再手写。那一步会改变**谁发出
通信**,超出"Ulysses/KCP 逻辑不变"的约束,所以没做。

现在这一步的价值是:CP 的契约变得可见、可测、与上游同形,将来换成 `local_map` 是替换函数体,
不是重构结构。**代价是它本身并不解锁 `spmd_types`** —— 手写集合通信跑在 `to_local()` 后的
plain tensor 上,spmd 类型检查大概率不认。这一点没有实测,不要当成已知结论。

## 验证标准

因为数值路径一行未动,**21 个 CP 格子应当与基线逐位一致**,不是"接近"。任何差异都说明
改写碰到了不该碰的东西。

### 靶子不能用 8-19 那份

第一反应是拿 `gate_logs/gate_58_2026-08-19_merged_percell.txt` 当基线。**那份对当前树已经作废**:
8-19 之后这棵树上有两个真改数值的提交 ——

* `e2b655593` 拆分的视觉塔根本没有 deferred backward(梯度变了)
* `740c41a1b` bubble 按预算放置多个 encode,不再是一个

拿它比,树差会被算成本次改写的效果。所以跑 **42 格**:先在当前树上**不带**改动重建基线
(`/workspace/mx_cp_before`),再带上跑一遍(`/workspace/mx_cp_after`),两者相比。

这条一般化:**基线的有效期到下一个改数值的提交为止**,而不是到下一次 gate 为止。引用一份
基线时必须同时写清它对应的 HEAD。

### 记账 bug:`FAIL (20/10 steps)` 是假报

基线首跑时每个格子都报 FAIL,数字是 20/10、40/10、80/10 —— **恰好是 rank 数乘步数**。
每个 rank 都打 loss 行,而 `run_cp_cells.sh` 用裸计数,gate 用的是 `uniq`。数值本身没问题。

`cmp_cp_cells.py` 改成**按 step 号去重**,没有沿用 `uniq`,理由有两条:

* `uniq` 折叠的是*相邻相同*。连续两步的 loss 与 grad_norm 都恰好落在同样的五位有效数字上
  虽罕见但真实存在,那会被静默吃掉一步,报成"步数不足";
* 按 step 号去重顺带能**断言各 rank 数值一致**。rank 失步正是 CP 才会有的 bug,而在
  `uniq` 方案里它完全隐形 —— 第一个 rank 的数字和基线对得上,整个跑看起来干干净净。

`run_cp_cells.sh` 自身的计数还没修(改写时脚本正在运行,bash 会边执行边读文件,中途改会损坏
执行)。结论以 `cmp_cp_cells.py` 为准。

### 匹配能证明什么,不能证明什么

两侧都是 stdout,只有五位有效数字。逐位相同**不能**排除更低位的差异 —— 要更强的结论得从
TensorBoard 记录取(上游 `scripts/loss_compare.py` 的做法)。这里选 stdout 是因为它与既有
gate 记录同源、可直接对照。

### 为什么改动侧跑 58 而不是 21

21 格是"带 CP 的格子",但**改动的爆炸半径不止于此**。两处落点:

* `apply_cp_kimi_k3` —— 只有 cp>1 才进,确实只碰 CP 格;
* `model.py::KimiDeltaAttention.__init__` 的 `contract_for_mode(self.cp_mode)` ——
  **每次建模型都跑**,与 CP 无关。

第二处是纯校验,按理不可能改数值。但"按理不可能"正是这个仓库里反复出事的推理形态,而证伪它
只需要跑一次本来就该跑的 gate。所以:基线 21 格 + 改动侧完整 58 格。gate 自己的 CP 格用的是
同一套 arm knobs/extras(`run_cp_cells.sh` 启动时断言了这一点没漂移),所以那 21 格与基线可直接比。

21 格 = 每臂 7 个 × 3 臂(text / mm_full / mm_lora):

    cp2  fsdp2_tp2_cp2  tp2_pp2_cp2  fsdp2_pp2_cp2
    ep2_fsdp2_tp2_cp2  ep2_fsdp2_pp2_cp2        (run13_flav.sh)
    cp4                                          (run_maxdeg.sh)

## 下次上游合并时会断的东西

`parallelize.py` 三处引用 `full_dtensor`,PR-4217 已删掉该后端。不属于本次改写范围,但合并时
必须一起处理。已经查清,修法是确定的,不必到时重新考古:

* `torchtitan/distributed/full_dtensor.py` 整个删除(167 行),`resolve_fsdp_mesh` 搬到
  `torchtitan/distributed/fsdp.py`,**签名逐字未变**,只多了一句
  `assert parallel_dims.spmd_backend == "spmd_types"`。
* 所以改两处:导入路径换成 `from torchtitan.distributed.fsdp import resolve_fsdp_mesh`;
  条件 `if parallelism.spmd_backend in ("full_dtensor", "spmd_types")` 收成
  `== "spmd_types"` —— 不收的话会撞上那句新 assert。
* `parallelize.py:1445` 注释里提到 full_dtensor,一并改。

## 实测:KCP 不是特有的不兼容项(2026-08-20)

`matrix_scripts/probe_spmd_types_cp.sh`,`{kcp, ulysses} x {partial_dtensor, spmd_types}`:

| | partial_dtensor | spmd_types |
|---|---|---|
| kcp | ok | **fail** |
| ulysses | ok | **fail** |

两者**以完全相同的错误失败**:

    ValueError: When dp_mesh_dims is provided, all parameters must be DTensors on the
    full SPMD mesh (e.g. via distribute_module). Got plain tensor for parameter 'weight'.

失败点在 `fully_shard` / `apply_fsdp` 的参数分片阶段。日志里有 `Applied CP`,但 **step 行为零 ——
前向没跑到**。

所以:

1. **先前"KCP 最可疑"的判断是错的方向。** Ulysses 一模一样地挂。
2. **阻塞在 CP 上游**:参数不是全 SPMD mesh 上的 DTensor。这正是那个已知缺口 —— 支持
   `spmd_types` 就是给这个模型做声明式参数分发(`sharding.py` 的另一半),不是补丁。
3. **KCP 到底有没有问题,这次没有答案。** 跑死在参数阶段,"逐 rank 顺序扫描能否过 spmd 类型检查
   和图捕获"仍然未知,只是被挡在参数转换后面。**不要把这次实测当成 KCP 已澄清。**

顺带解释了 gate 为什么必须 pin `partial_dtensor`:不是保守,是当前树在 `spmd_types` 下根本起不来,
且与 CP 无关。

## 还没做的一步

gate 的 `GATE_EXTRA` 仍 pin 着 `--parallelism.spmd_backend partial_dtensor`。上游的
`validate_cp_backend` 已规定 CP 必须配 `spmd_types`,所以合并之后这个 pin 与 CP 不能共存。
解开它需要的正是"完整声明式"那一步(边界发通信),也就是本文开头说的没做的那半。

顺带:`matrix_scripts/run_cp_cells.sh` 的 `BASELINE` 默认值仍指向 8-19 那份作废基线,
调用时必须显式覆盖。
