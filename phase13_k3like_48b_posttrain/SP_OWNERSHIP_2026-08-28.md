# SP 落地与 TP-SP×CP 序列所有权契约(2026-08-28/29,本地 8×4090)

树 `k3_on_4025`,提交链:`ab2b8d42a`(SP 一阶段)、`9637a03dd`(契约 +
缝一)、`f7033cc75`(vit 测试 init 修)、`ae5fcd3f9`(缝二/缝三 + 线程钉)。
判据 `EVIDENCE_METHOD_2026-08-25.md`(单 seed 一表、warm 后测、s1/s3/s10、
step-2 准则)。raw:`/workspace/mx3_sp_table_*`、`mx3_sp_nosp_*`、
`mx3_sp_full_*`、`mx3_sp_text_*`、`mx3_sp_pp*`。
**SP 工作只在集成树,零 PR 分支泄漏**(TP/QB/QAT/LoRA/EP 各 PR 分支均不含)。

## 一、契约(核心改动,写入 model.py 注释)

TP-SP 与 CP 都想切 token 轴,声明式分片下二者的边界:

- **CP 拥有外层切分**:数据级、带外(out-of-band)——loss/positions/数据
  管线看到的就是 cp-local 流,声明式重分布不跨它;
- **TP-SP 拥有内层切分**:在 cp-local 流上的带内 Shard(0)(tp 轴),模块
  边界 gather/scatter 只在这一层;
- **交汇点**是 MLA/KDA 的 in_dst 声明 `(tp: Replicate, cp: Shard(0))`:
  进注意力时 gather 掉 TP 份、保留 CP 份交给 Ulysses/KCP;
- **KDA 例外**:delta 递归消费整条序列 ⇒ 模块边界 gather 进/slice 出,
  SP 的激活省显存不延伸进 KDA 体内;
- **缝的类别**:所有"token 索引直接下标、且数据来自带外流"的代码
  (mm splice、positions)——带内声明看不见它们,须逐处审。

## 二、缝链(三道,全部落在预告的"缝的类别"里)

**缝一:非 PP mm splice**(9637a03dd)。tp2sp×cp2 首探针在
`_prepare_multimodal_embeds` 崩(256 cp-local vs 128 tp-sp local,dim 0):
塔输出按带外 token 索引 masked_scatter 进文本流,而文本流此时带着带内
Shard(0)。修法 = **splice 前 redistribute 回 cp-local 全量、splice 后回 SP
布局**(双向可微)。弃选项:按 tp rank 切 mask 切塔输出——会给"各 tp rank
复制的塔"制造不相交的部分梯度,tp 轴上没有任何归约把它们合回去,静默错梯度。

**缝二:跨段 mm splice**(ae5fcd3f9,tp2×pp2vp2 揭出)。
`dep_vision_stage.py` 的跨段 splice 注释自陈假设"TP 下文本流 local 即全量"
——Replicate 时代成立;SP 下段边界 local 是半长 Shard(0) 切片而 mask/塔特征
全长(expand 256 vs 512 当场崩)。修法与缝一完全同构。

**缝三:keepalive 染标**(ae5fcd3f9;这道最隐蔽,值不变、只有布局被改)。
AttnRes 跨段 adapter 用 `0.0 * recv.sum()` 保梯度图边;SP 下 delta 载体是
Shard(0),全局 sum 得 **Partial(sum) 标量**,加到 Replicate 的末段输出上
把 loss 输入整体染成 P——local 数值一个没变。定位链:loss 入口布局探针
(首 mb R、次 mb P;R 系 metadata-inference 直调路径)→ 末 norm 出口探针
反证(每 mb 都出 R)→ 逐层排除调度/stage/根模块 → keepalive。修法:零值
touch 先 redistribute 回 Replicate 再加(标量 allreduce,数值自由)。
**先试并被反噬的方案**:对 local 分片求和——backward 里 autograd 引擎给
标量边递回 DTensor 梯度,`to_local` 的反向拿它去 `from_local` 双重包装,
当场抛错。此教训与 house rule"to_local 必须显式 grad_placements"同源。

**非缝一则**:tp2×pp2vp2×fsdp2×ep2 的首败是上游新加的步整除检查
(dp2×8 微批×512=8192>4096),微批降 256 解决,几何注记于表。

## 三、数值表(15 格,老树口径;同 seed,warm+measure)

mm = kimi_k3_debugmodel;text = kimi_k3_debugmodel_32l。PP 格:
Interleaved1F1B,8 微批,layers-per-stage 6(vp2)。

| 格 | GPU | s1 | s3 | s10 | 峰值显存 | tps |
|---|---|---|---|---|---|---|
| dp1(基线) | 1 | 12.45537 | 7.89917 | 4.01159 | 11.94GiB | 547 |
| tp2(SP 开,默认) | 2 | 12.43649 | 7.65014 | 3.95362 | 7.09GiB | 147 |
| tp2(SP 关,`--no-enable-sequence-parallel`) | 2 | 12.45614 | 7.79380 | 4.07169 | 7.11GiB | 170 |
| tp4 | 4 | 12.42574 | 7.53798 | 4.05860 | 4.27GiB | 72 |
| fsdp2×tp2 | 4 | 12.49407 | 7.33544 | 3.62537 | 4.25GiB | 107 |
| fsdp2×tp4 | 8 | 12.47875 | 7.30622 | 3.67881 | 2.72GiB | 62 |
| tp2sp×cp2 | 4 | 12.43947 | 7.70929 | 4.11745 | 4.09GiB | 48 |
| fsdp2×tp2sp×cp2 | 8 | 12.49234 | 7.25718 | 3.60453 | 2.94GiB | 47 |
| fsdp2×tp2×cp2(SP 关,叠加态消融) | 8 | 12.49554 | 7.39010 | 3.59432 | 2.96GiB | 52 |
| fsdp2×tp2×ep2 | 4 | 12.49511 | 7.26491 | 3.50688 | 4.32GiB | 104 |
| fsdp4×tp2×ep4 | 8 | 12.44600 | 6.43510 | 3.50401 | 3.12GiB | 111 |
| tp2×cp2×fsdp2×ep2(四联岛) | 8 | 12.51709 | 7.30760 | 3.62234 | 2.94GiB | 52 |
| tp2×pp2vp2 | 4 | 12.45594 | 7.69221 | 3.95175 | 5.38GiB | 102 |
| tp2×pp2vp2×fsdp2×ep2(微批 256) | 8 | 12.46893 | 6.77196 | 3.30495 | 3.88GiB | 38 |
| text32l×tp2sp×cp2 | 4 | 12.41179 | 7.55171 | 3.94268 | 4.80GiB | 38 |

设计性守卫沿用老表:`tp2×cp2×pp2vp2`(跨段塔 + CP)不入跑格。

判读:

1. 15 格全 rc=0、seed 门通过,s1 全在惯常跨并行带内(±0.03;SP 改变归约
   次序,不与 dp1 位同);SP×EP 族、SP×PP 族、四联岛、text 契约格全绿。
   fsdp4×tp2×ep4 的 s3 偏快系 dp4 数据切分次序差异,s10 与 fsdp2×tp2×ep2
   同带(3.504 vs 3.507)。
2. **debug 规模下 SP 无显存收益、有速度代价**(7.09≈7.11GiB;147 vs
   170 tps;叠加态 SP 开/关同样 2.94≈2.96GiB、47≈52 tps):seq=512、窄
   dim 时被省的 norm/激活份额可忽略,而模块边界 redistribute 是净开销。
   SP 的收益主张属于长序列/大 dim 场景,本表不为其背书,只证明**正确性
   与可叠加性**。
3. **标注**:此前全部 TP 证据(TP_FULL_MATRIX、各 PR 表)均为 SP-off 时代
   ——当时 K3 尚未消费 core 旗标。本次起 `parallelism.enable_sequence_parallel`
   (core 默认 True)被 K3 消费,**tp 格默认即 SP-on**;旧表不失效(各表
   单 seed 自带基线,不跨表比绝对值),新表内 SP 关格即旧行为的对照。
4. 测试:`kimi_k3/tests` 全目录 43 通过 + 186 子测试(空载终验)。

## 四、vit 测试 flake 归因更正(双因子,替代此前"脏树污染"结论)

此前把 `test_vit_stage_shares` 的偶发失败归因于 stash 残渣污染——**该结论
错误**(干净树、无改动同样复现,失败测试随机漂移)。真因两条,均已修:

1. **未初始化权重 × 分配器复用**(f7033cc75):`_tower()` 只 `build()`
   从不 `init_states()`,`pos_embed` 等延迟初始化参数一直是 `torch.empty`
   裸内存。两条对比路径共用同套(垃圾)权重,垃圾有限则位同恒过;qlora
   打包测试(`test_experts_pack_at_build_and_dequant_property`)释放的
   uint8 缓冲被复用后解出 NaN,`rtol=0,atol=0`(equal_nan=False)当场爆。
   前置文件依赖 = 分配器 binning 巧合(4/6 前置文件触发)。二分链:全目录
   → 文件前缀二分 → 成对扫描 → 单测定位。
2. **宿主负载下的线程分块漂移**(ae5fcd3f9):init 修后失败仍偶发,且只在
   与 GPU 矩阵并行时出现——oneDNN 按负载调整线程分块,归约次序变则位同
   破。修法:比较段内钉单线程(addCleanup 恢复)。

## 五、未尽

- positions 等其余 token 索引位点在探针与全测试下未触发;若后续 flavor
  (chunked loss×MTP 等)引入新的带外索引消费者,按 §一"缝的类别"逐处审。
- 长序列规模的 SP 显存/速度收益表(需要放大 seq 的专门格),与 48B 形状
  验证一并排期。
