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
