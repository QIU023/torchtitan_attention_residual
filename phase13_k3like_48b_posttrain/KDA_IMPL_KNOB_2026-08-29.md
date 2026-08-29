# KDA impl 旋钮分支证据(2026-08-29,本地 8×5060Ti,SM120/cap 12.0)

分支 `k3_kda_impl`,tip `3b34e3068`(基 upstream/main `13da2d77c`;trailer
已按上游分支规则清除)。body:`Raising_PRs/PR_K3_PARALLELISM/PR_BODY_KDA_IMPL.md`。
raw:`raw_kda_ref_0829/`(results.txt + 两格 measure 全量 gz + 关键行摘录,
摘录含 auto 落 reference 的 info 行:"CUDA capability (12, 0) has no fused
kernel; using Attention Gym's reference implementation")。

背景:上游 `d208df86f`(PR-4351)将 K3 KDA 换到 Attention Gym fused 内核,
`impl="fused"` 硬编码 + SM100/SM103 之外无条件 raise——K3 成为仓库唯一
"模型本体只跑 DC Blackwell"的例子(其余模型的 SM100 门全在可选量化/
cudagraph 层)。旋钮 = `KDAKernel.Config.impl` auto/fused/reference,
默认 auto;SM100/103 上行为与数值零变化。

## 证据链(全部在 SM120 上)

1. lint:pre-commit 全绿(pyrefly 0 errors);
2. 上游自家 oracle 经 reference 路径通过:`test_kda_attention`(varlen 拼接
   vs 独立文档,前向+梯度)+ `test_kimi_k3::…matches_recurrent_reference`
   (内核 vs FP32 顺序递归)——4 passed(原先非 B200 全 skip);
3. debugmodel 整模训练(单 seed,warm+measure,s1/s3/s10):

| 格 | s1 | s3 | s10 |
|---|---|---|---|
| dp1 | 12.57037 | 7.56620 | 3.94650 |
| dp2 | 12.51296 | 7.56612 | 3.27074 |

## 后续(不入此 PR)

- KCP 从 fla 前缀扫描迁至 attn_gym(`chunk_kda` 有 `initial_state`/
  `output_final_state` 状态口,LASP 式跨 rank 传递起步);conv halo 同迁;
- 集成树 rebase 跨 `d208df86f` 时的 KDA 声明重瞄准(InnerKDA/KDAKernel 新
  Config 树)。
