# TP/SP 草稿分支证据 + 三分支夜检(2026-08-29,本地 8×5060Ti)

raw:`raw_tpsp_0829/`。判据 `EVIDENCE_METHOD_2026-08-25.md`。

## 一、k3_tp_sp 六格(head `ea0212643`,基 30eb5e50,pre-#4351 本机可跑)

SP 移植系手工端口(cherry-pick 因分支把声明折进 sharding.py 而全冲突):
`set_tensor_parallel_sharding_config(config, *, enable_sp)` 全线打通、
MLA/KDA 模块边界 SP 声明、`rowwise(output_sp)`、mm splice 回全量缝修
(适配 `scatter_vision_embeds` 形态)、消费 `parallelism.enable_sequence_parallel`。

| 格 | GPU | s1 | s3 | s10 | 峰值显存 | tps |
|---|---|---|---|---|---|---|
| dp1 | 1 | 12.58962 | 8.12642 | 3.95057 | 11.94GiB | 540 |
| tp2 SP开 | 2 | 12.58262 | 8.20421 | 3.97382 | 7.09GiB | 147 |
| tp2 SP关 | 2 | 12.61339 | 8.64604 | 3.88164 | 7.11GiB | 167 |
| fsdp2×tp2 | 4 | 12.58591 | 7.79146 | 3.51720 | 4.25GiB | 104 |
| tp4 | 4 | 12.59771 | 8.56692 | 3.98220 | 4.27GiB | 74 |
| fsdp2×tp4 | 8 | 12.61042 | 8.04123 | 3.57420 | 2.72GiB | 61 |

六格全 seed 门过;判读同集成树 SP 章(debug 规模 SP 无显存收益有速度
代价;正确性与可叠加性主张)。表已填入 `PR_BODY_TP_SP.md`,body 无占位。

## 二、EP head 终态位同验证(261fe3290,body 精确旗标 8192/256)

be47793 之后的四个提交(注释中立化、flavor 删除、清理、trailer 清洗 +
两轮污染退回)全为非计算改动。验证:dp2 与 ep2×fsdp2 两格六个数字与
`PR_BODY_EP_head_261fe329.md` 主表**逐位相同**(12.58904/7.61204/3.33502;
12.59108/7.59337/3.29070)——历史手术零数值影响,EP 线收口。

## 三、CP/PP 重做 squash tip 首跑

k3_cp(1b4eba328)与 k3_pp(f589f57f)系以 review tip 内容重做的单
squash(基对齐 30eb5e50),此前未执行过。冒烟:

| 分支 | 格 | s1 | s3 | s10 |
|---|---|---|---|---|
| k3_cp | dp1×cp2 | 12.59946 | 7.95798 | 3.81137 |
| k3_pp | dp1×pp2vp2 | 12.58962 | 8.05935 | 4.08061 |

pp 格 s1 与本批 dp1 位同(12.58962)——管线切分不动 step-1 损失的预期
性质,顺带验证 squash 未损内容。

## 附:Ulysses 文档 mask 修复(2026-08-29 晚,cp_review1 @a1a814073)

回应 PR-4313 reviewer 第二条(MLA 半边):CP 路原为 causal-only 重建 +
上下文窗守卫拒多文档流;修为 all-gather 连续 positions 份、按非 CP 同款
`causal×document` mod 建全序列 mask,守卫/缓存/死参数整链拆除。验证两层:
2-rank gloo 边界单测(三文档,一界压切口、一界在 shard 内;causal-only
必挂的两处拒绝断言)绿;GPU 回归三格(单文档 debug 数据,预期无扰):
dp1 12.58962/8.12642/3.95057、cp2 12.50615/8.09824/4.02580、
fsdp2×cp2 12.55773/7.94068/3.55172,同带。KDA 半边(cu_seqlens/状态重
置)归 lane A rebase 与 PR-4347 合流。raw:`raw_tpsp_0829/cpmask_results.txt`。

## 附二:mask 修复后的 CP body 双矩阵(2026-08-30,head a1a814073)

**归因实锤**:positions 探针显示 1024 折叠流含**两篇文档**(zeros=2)——
默认 mm 数据在此几何下打包;旧 causal-only 让 doc2 看见 doc1,而 dp1 一直
用文档 mask;旧守卫(仅 T>窗才 raise)在 T=窗+双文档时恰不触发。故新表
**非 CP 行与旧 body 表逐位相同、CP 行集体移动 = 修复生效**的直接证据。

主表(16384/1024/1024,七格):dp1 12.60544/7.30226/3.22742;
cp2 12.53996/7.04577/3.50511;cp4 12.52432/6.89080/3.49263;
cp8 12.53711/7.52113/3.49416;dp2 12.58193/7.44923/3.32128;
dp2×cp2 12.57299/7.50029/3.35914;dp2×cp4 12.53546/7.32970/3.38800。

4135 伴表:dp1 s10→3.23705;cp2 7.10562/3.38411;cp4 7.29175/3.36449;
cp8 s10→3.49337;dp2 s10→3.32140;dp2×cp2 三步位同;dp2×cp4 s10→3.39451。
纹样与旧伴表判读一致(grad-norm 分组的 bf16 舍入敏感格移动)。

KDA 半边(KCP cu_seqlens=[0,total])有意不修:非 CP 的 KDA 同样不重置
(#4347 未合并,全模型无 KDA packing),KCP 与单卡行为一致,边界重置随
#4347 全模型落地。cp2 同 seed 复跑通过后此表替换 PR body 主表。
raw:`raw_tpsp_0829/cpbody{,4135}_results.txt`。
