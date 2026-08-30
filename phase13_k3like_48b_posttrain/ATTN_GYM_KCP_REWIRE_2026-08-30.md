# KCP 换线到 attention-gym:七格矩阵在新依赖下的判定(2026-08-30,8×5060Ti)

cp_review1 新尖 `837a66235`:`conv_with_halo` 与 `build_kcp_context` 的两个 CP
import 从裸 fla(`fla.modules.conv.cp.ops` / `fla.ops.cp.context`)换到
`attn_gym.linear.kda.fla_cp`(meta-pytorch/attention-gym PR 421 的移植层),
parallelize 的 wiring 检查同步改名。fla 仍是底层内核;wrapper 预期为纯透传
(conv 权重 squeeze ≡ 原 rearrange)。

## 判定:六格逐位,cp4 定性为既有非确定格

主矩阵(`mx3_attngym_kcp_0830_060114`)+ CP 补跑
(`mx3_attngym_kcp_cp_0830_122323`),协议同 CP body(seed 42、determinstic、
双遍读第二遍、s1/s3/s10):

| cell | s1 | s3 | s10 | 对 body 主表 |
|---|---|---|---|---|
| dp1 | 12.60544 | 7.30226 | 3.22742 | 逐位 |
| cp2 | 12.53996 | 7.04577 | 3.50511 | 逐位 |
| cp4 | 12.53406 | 7.24227 | 3.35766 | 见下 |
| cp8 | 12.53711 | 7.52113 | 3.49416 | 逐位 |
| dp2 | 12.58193 | 7.44923 | 3.32128 | 逐位 |
| dp2×cp2 | 12.57299 | 7.50029 | 3.35914 | 逐位 |
| dp2×cp4 | 12.53546 | 7.32970 | 3.38800 | 逐位 |

## cp4:按启动非确定,与本次换线无关

- 同一次调用内,cp4 的 warm 遍 s1=**12.52432**(= body 主表值)、measure 遍
  s1=**12.53406**(= body fp32 表值)——同 seed、同 checkpoint、背靠背分叉。
- 这回溯解释了 body 两张表 cp4 在 **s1** 即不同的旧现象:grad-norm 精度不可能
  影响第一步前向,真因是该格前向自身按进程启动漂移(triton autotune 选配)。
- 历史横断面(mask 修复后可比三次):s1 ∈ {12.52432, 12.53406} 两吸引子。
- 判定依据"根因必须解释未失败的格":六格(含同为纯 CP 的 cp2/cp8)逐位,
  wrapper 透传成立;cp4 归格子自身。

## 事故记录:两次假信号

1. venv 里装着 PyPI 的 attn-gym 0.0.5(无 `linear.kda`),而 import 探测在
   attention-gym 仓库目录里跑,cwd 解析假阳性 → 首轮 CP 三格在 wiring 检查处
   全灭。教训:依赖探测必须在中性 cwd 验证 `__file__`。
2. "等首轮退出再补跑"的链式后台任务用 `pgrep -f mx3.sh` 做等待条件,匹配到
   自己命令串,循环永真,8 卡空转五小时(pkill 教训的 pgrep 版,记忆已扩写)。

## 关联

- PR_BODY_CP.md 已加两段 review round(attention-gym 依赖 + 逐位结论句)。
- attention-gym 侧:PR 421(用户已发,head a8c4423 = 819370f + 上游 main 合并,
  我们四个文件逐字节同)。

## 附:4135(fp32 grad-norm)矩阵在新依赖下 7/7 逐位复现

`mx3_attngym_kcp_4135_0830_134302`,树 = 837a66235 + cherry-pick 578aee746。
七格与依赖变更前的旧 4135 表**逐位相同**(cp4 本次恰落同吸引子 12.53406,
使 s3/s10 也可比且相同)。body 的两张旧表已按用户指示删除,新表
(bf16 主表 + fp32 表,均测于本头)已写入 PR_BODY_CP.md。
