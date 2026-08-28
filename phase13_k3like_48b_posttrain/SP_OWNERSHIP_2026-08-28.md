# SP 落地与 TP-SP×CP 序列所有权契约(2026-08-28,本地 8×4090)

树 `k3_on_4025`,两个提交:`ab2b8d42a`(SP 一阶段)、`9637a03dd`(契约 +
第一道缝)。判据 `EVIDENCE_METHOD_2026-08-25.md`(单 seed 一表、warm 后测、
s1/s3/s10、step-2 准则)。raw:`/workspace/mx3_sp_table_*`、`mx3_sp_nosp_*`。
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

## 二、第一道缝:mm splice(9637a03dd)

tp2sp×cp2 首探针在 `_prepare_multimodal_embeds` 崩(256 cp-local vs 128
tp-sp local,dim 0):塔输出按带外 token 索引 masked_scatter 进文本流,而
文本流此时带着带内 Shard(0)。修法 = **splice 前 redistribute 回 cp-local
全量、splice 后回 SP 布局**(双向可微)。弃选项:按 tp rank 切 mask 切塔
输出——会给"各 tp rank 复制的塔"制造不相交的部分梯度,tp 轴上没有任何
归约把它们合回去,静默错梯度。

## 三、数值表(六格,同 seed,warm+measure)

| 格 | GPU | s1 | s3 | s10 | 峰值显存 | tps |
|---|---|---|---|---|---|---|
| dp1(基线) | 1 | 12.45537 | 7.89917 | 4.01159 | 11.94GiB | 547 |
| tp2(SP 开,默认) | 2 | 12.43649 | 7.65014 | 3.95362 | 7.09GiB | 147 |
| tp2(SP 关,`--parallelism.no-enable-sequence-parallel`) | 2 | 12.45614 | 7.79380 | 4.07169 | 7.11GiB | 170 |
| fsdp2×tp2(SP 开) | 4 | 12.49407 | 7.33544 | 3.62537 | 4.25GiB | 107 |
| tp2sp×cp2 | 4 | 12.43947 | 7.70929 | 4.11745 | 4.09GiB | 48 |
| fsdp2×tp2sp×cp2 | 8 | 12.49234 | 7.25718 | 3.60453 | 2.94GiB | 47 |

判读:

1. 六格全 rc=0、seed 门通过,s1 全在惯常跨并行带内(±0.02;SP 改变归约
   次序,不与 dp1 位同);叠加双格(×cp2、×fsdp2×cp2)收敛形态正常。
2. **debug 规模下 SP 无显存收益、有速度代价**(7.09≈7.11GiB;147 vs
   170 tps):seq=512、窄 dim 时被省的 norm/激活份额可忽略,而模块边界
   redistribute 是净开销。SP 的收益主张属于长序列/大 dim 场景,本表不为
   其背书,只证明**正确性与可叠加性**。
3. **标注**:此前全部 TP 证据(TP_FULL_MATRIX、各 PR 表)均为 SP-off 时代
   ——当时 K3 尚未消费 core 旗标。本次起 `parallelism.enable_sequence_parallel`
   (core 默认 True)被 K3 消费,**tp 格默认即 SP-on**;旧表不失效(各表
   单 seed 自带基线,不跨表比绝对值),新表内 SP 关格即旧行为的对照。
4. 测试:`kimi_k3/tests` 全目录 43 通过 + 186 子测试。(过程记事:首轮
   两例 vit 失败系陈旧 stash 残渣污染工作树,清树后确定性全绿;肇事
   stash 已 drop,残件归档 `raw_h100nvl_0828/verl_relic_parallelize_fragment.py`。)

## 四、未尽

- positions 等其余 token 索引位点在 tp2sp×cp2 探针与全测试下未触发;若
  后续 flavor(chunked loss、MTP)引入新的带外索引消费者,按 §一"缝的
  类别"逐处审。
- 长序列规模的 SP 显存/速度收益表(需要放大 seq 的专门格),与 48B 形状
  验证一并排期。
