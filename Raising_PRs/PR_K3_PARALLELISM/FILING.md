# K3 并行 PR:三份文本侧正文

这个目录只放**可直接粘贴的 PR 正文**,纯英文,无小标题、无粗体结构。
证据表进正文(用户 2026-08-25 要求:step 1 / step 3 / step 10)。
判断、结果原始记录、方法学、状态汇总一律在 logbook,不在这里。

| 正文 | 分支(fork `QIU023/torchtitan`) | 基线 | 范围 |
|---|---|---|---|
| `PR_BODY_PP.md` | `k3_pp_text` | `upstream/main` | 文本解码器 PP + AttnRes 跨 stage |
| `PR_BODY_CP.md` | `k3_cp_text` | `upstream/main` | 文本解码器 CP:MLA Ulysses + KDA KCP |
| `PR_BODY_EP.md` | `k3_ep` | `upstream/main` | EP 声明式接线,无 MoonEP |

三份正文都不谈视觉塔:DEP(塔独占 stage)与 dynamic CP(大图沿 patch 维切分)
留在 `k3_pp` / `k3_cp` 两个多模态分支上,正文里各有一句"后续在多模态路径提"。
分支侧已验证:`k3_pp_text` 0 处 DEP,`k3_cp_text` 0 处 dynamic CP。

证据在 logbook:

* `phase13_k3like_48b_posttrain/PP_TEXT_EVIDENCE.md`
* `phase13_k3like_48b_posttrain/CP_TEXT_EVIDENCE.md`
* `phase13_k3like_48b_posttrain/EP_TEXT_EVIDENCE.md`
* `phase13_k3like_48b_posttrain/RAISE_READINESS_2026-08-24.md`(合格度严判)
* `phase13_k3like_48b_posttrain/PARALLEL_PR_STATUS_2026-08-24.md`(分支/测试状态)

方法学(每张表都成立):同一个 seed checkpoint 被每格载入;每格先跑一次丢弃的
同配置预热(冷/热 inductor 缓存差 ~7e-3,大于真实跨格差);每个计量格断言日志出现
`Loading the checkpoint from`,否则标 ASSERT-SEED-FAIL。矩阵跑在**将要提交的那个
分支的 worktree** 上,不是集成树 —— 数字属于 reviewer 看到的那份 diff。

未经用户逐字确认,不提交任何一份。
