# Quantile Balancing for MoE load balance

按 `CLAUDE.md` 的 PR-text 规则写:正文无小标题、无粗体结构、无证据表格,
单行段落以便逐字复制。证据留在本文件的"支撑材料"一节,不进 PR body。

目标 issue:4272 的 P0 条目 "Implement Quantile Balancing",两个子项分别是
"Support distributed histogram aggregation and gradient accumulation" 与
"Save and restore its state through DCP"。

**这个改动不依赖 4025**,`quantile_balance.py` 的 import 只有 `torch`,
认的是 `torchtitan.models.common.moe.MoE` 和 core 已有的 `expert_bias_E` buffer。
任何用 core MoE 的模型都能用。

--- PASTE BEGIN ---

Adds Quantile Balancing as an alternative to the sign-rule bias update for MoE load balance. It solves for each expert's bias from the distribution of routing margins rather than nudging it by a fixed step, so a layer converges in one optimizer step instead of over many, and the step size stops being a hyperparameter.

The rule is the one in Kimi K3's report, section 2.2. Routing takes Top-(k+1) so the (k+1)-th score is the cutoff an expert must beat to enter a token's Top-k; the margin of expert e on token t is its score minus that cutoff. Solving for the bias that puts each expert at the target load is then a quantile of the margin distribution, which a histogram approximates to whatever resolution num_bins buys.

It composes with data and context parallel by summing the histograms. Integer counts are additive, so one SUM all-reduce over the loss mesh reconstructs the whole-batch margin distribution regardless of how tokens were sharded. The reduction is one collective for all layers rather than one per layer -- every histogram has the same shape and dtype, so they stack, and at K3's 92 layers the difference is 92 blocking round trips per optimizer step against one.

The loss mesh deliberately excludes TP. Under tensor parallel the router's scores are replicated, so every TP rank holds the same tokens and summing over that axis would multiply the histogram by the TP degree. Measured by all-gathering the pre-reduce counts: across the loss group they differ (34765 of 81920 bins at dp2, 32928 at dp4), so the reduction aggregates distinct data.

State survives checkpointing without new machinery. The solved bias lands in `expert_bias_E`, which core already registers with `persistent=True`, and the per-step histograms are cleared after each solve, so there is nothing else to persist. An uninterrupted run and one resumed from step 5 produce identical losses and gradient norms for steps 6 through 10, and `expert_bias_E` comes back with the same values.

Memory is one (E, num_bins) int32 histogram per MoE layer, about 169 MiB at 896 experts by 512 bins by 92 layers. num_bins trades quantile resolution against that.

Offline, expert-load coefficient of variation drops from 0.607 to 0.053.

--- PASTE END ---

## 支撑材料(不进 PR body)

| 声称 | 证据 |
|---|---|
| 聚合是真实的、TP 被正确排除 | 全 gather 前置计数逐 bin 比对:dp2 34765/81920 不同、dp4 32928/81920 不同 |
| DCP 存取 | 不中断与 resume 的 step 6-10 逐位相同;`expert_bias_E` @ step-10 min=-0.025539 max=0.017409 两侧一致 |
| TP 下可用 | dp2+tp2、ep2+dp2+tp2 各 10 步;dp4 数值不变 |
| 收敛 | `QUANTILE_BALANCING.md` 的离线定点实验 cv 0.607 -> 0.053 |
| 单测 | `tests/test_quantile_balance.py` 273 行 |

## 提交前要做的

1. **搬文件**:`torchtitan/models/kimi_k3/quantile_balance.py` -> `torchtitan/components/` 或
   `torchtitan/distributed/`。它不属于任何一个模型 —— 零 K3 依赖,认的是 core 的 `MoE`。
   放在 `models/kimi_k3/` 下只是因为它是在做 K3 时写的。
2. **测试跟着搬**,并去掉测试里对 K3 flavor 的依赖(如果有)。
3. `register_quantile_balancing` 的 `post_optimizer_build_fn` 用法要在 docstring 里给出,
   现在只有 K3 的 flavor 在用。

## 一处要主动说明的

TP 被排除在 loss mesh 之外,是因为**当前 EP 关闭时** router 的 scores 是 Replicate。
EP 打开后 token 会被切分,那时直方图必须在 token 被切的那条轴上归约 ——
代码注释写明了这个边界,`ep2+dp2+tp2` 也跑过,但**"EP 下聚合正确"这一条我们只验了能跑,
没有验计数的正确性**。PR 里不要声称它。
