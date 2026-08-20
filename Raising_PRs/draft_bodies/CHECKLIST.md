# Draft PR 开箱清单(2026-08-20 更新)

四个分支已推到 `QIU023/torchtitan`,**每个的提交信息里引用数都是 0**,所以开 PR 不会在
pytorch/torchtitan 触发交叉引用事件。这是此前的硬阻塞:原来的 `k3_pr_{tp,ep,base}` 各带 383+
个提交、其中 14 条含裸 `#NNNN`,开 PR 会逐条触发。

重建方式:`git checkout upstream/main` 后把该分支的树作为**一个新提交**加上去。上游提交是父
history,对象和 message 一字未动 —— 没有重写任何人的提交。

base 一律选 `pytorch/torchtitan:main`,**勾 draft**,标题带 `[do not review yet]`。
body 全文在同目录 `.md` 里,直接粘。

| 分支 | 内容 | 标题 | body |
| --- | --- | --- | --- |
| `k3_pr_classified` | 全树 DO NOT MERGE,15 个按内容切的提交 | `[DO NOT MERGE] Kimi K3: full tree, pending rebase onto the reference-model PR` | `../PR29_torchtitan_kimi_k3_full_tree_draft/PR.md` 的 PASTE |
| `k3_pr_tp_clean` | 模型 + TP,CP/EP 各一句 raise | `[do not review yet] Kimi K3: tensor parallelism, including the KDA path` | `k3_pr_tp.md` |
| `k3_pr_ep_clean` | 模型 + EP,TP/CP 各一句 raise | `[do not review yet] Kimi K3: expert parallelism and the grouped-GEMM expert layout` | `k3_pr_ep.md` |
| `k3_pr_base_clean` | 模型 + PP adapter,三个轴都 raise | `[do not review yet] Kimi K3: pipeline parallelism with Block Attention Residuals` | `k3_pr_base.md` |

compare 链接:

* https://github.com/pytorch/torchtitan/compare/main...QIU023:torchtitan:k3_pr_classified?expand=1
* https://github.com/pytorch/torchtitan/compare/main...QIU023:torchtitan:k3_pr_tp_clean?expand=1
* https://github.com/pytorch/torchtitan/compare/main...QIU023:torchtitan:k3_pr_ep_clean?expand=1
* https://github.com/pytorch/torchtitan/compare/main...QIU023:torchtitan:k3_pr_base_clean?expand=1

三个轴分支各含完整模型,所以 diff 大量重叠 —— body 里已写明,免得被读成三份重复工作。
`parallelize.py` 的行数是它们真正的区别:TP 1422 / EP 646 / PP(base)482。

**发之前 body 要你逐字过一遍**(pytorch 的 AI policy 要求),我不自行提交。
