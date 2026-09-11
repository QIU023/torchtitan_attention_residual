# TP/SP (PR 4499) 返工清单 — 给 GPU 盒子(2026-09-10)

审计对象:`tp_sp_on_main` = `8a9c129d7`(review 分支;PR 分支 `k3_tp_sp` = `ce10692c1` 未动),
基 main `ac10ca48f`(4527 合入后)。body 文件:`Raising_PRs/PR_K3_PARALLELISM/PR_BODY_TP_SP_ON_4527.md`
(其余 `PR_BODY_TP.md` / `PR_BODY_TP_SP.md` / `PR_BODY_TP_SP_SPMD.md` 是旧版,做完后删或标 superseded)。

已在 `8a9c129d7` 上做掉的:`preprocess_inputs` 的 `# pyrefly: ignore [bad-override]` 删除
(签名与 `Decoder.preprocess_inputs` 一致,main 上的同名 override 也没有这条);四个带 cp 轴的
声明函数头各加一行 `# TODO: rebase after upstream CP merge`。

## 1. core `distributed/utils.py`(`_total_norm_by_mesh` / `_clip_grads_by_mesh`,+58/−17)— 先验证,大概率撤

依据:`set_tensor_parallel_sharding_config(..., spmd_types=...)` 在 `spmd_types` 下给塔声明
`{DP: R, CP: R, TP: I}`,所有参数都带 tp 轴;"复制的塔在 `(fsdp,)`" 那条记录
(`TP_FULL_MATRIX_2026-08-27.md:54`)是 `partial_dtensor` 时期测的。老树的解法本来就不改 core:
把复制模块在 tp 上声明 Replicate(`TP_DTENSOR_CONSTRAINTS.md` 的 NoParallel 段)。

做:
1. 单独 revert `8a28b4bca`,跑 K3 tp2(dp2 x tp2,SP on/off)各一次,两个后端各一次。
2. 预期:`spmd_types` 通过 → core 改动撤掉;`partial_dtensor` 在 `clip_grad_norm_` 报 mixed-mesh。
3. `partial_dtensor` 的处理二选一:给塔/projector 补 tp 维 Replicate 的声明;或不支持
   `partial_dtensor` + TP,`parallelize_kimi_k3` 明确 raise。不要为它改 core。
4. 若实验推翻预期(`spmd_types` 下也 mixed-mesh),先把是哪个模块落在 `(fsdp,)` 上打印出来,
   再决定;顺带删掉 `clip_grad_norm_` 里已成死码的 `if isinstance(total_norm, DTensor)` 段和它上面的 NOTE。

## 2. `kimi_k2_7/vision_encoder.py`(+3/−6)— 拆成独立小 PR

`_compute_learned_pos_embeds` / `_compute_2d_rope_cache` 由 K2.7 与 K3 共用,改动是"只 mutate dp 轴,
tp 轴保留表的声明(`I`)"。K2.7 自己也声明 `pos_embed` / `inv_freq` 在 tp 上为 `I`,同样的 src 检查在
tp>1 开类型检查时也会报——只是 CI 没有 K2.7 的 TP 格,没人撞见。

做:单独一个 PR(两个函数、一句话说明、K2.7 与 K3 各一个 tp2 类型检查格);4499 声明依赖它。
如果不想再开 PR,至少跑一次 K2.7 tp2 + 类型检查,结果写进 body 的 Design。

## 3. K3 的 TP CI 格 — 现在一个都没有

CI 里 K3 只有 `kimi_k3_debugmodel_mm_fsdp2`(b200,dp2)。加一个 `kimi_k3_debugmodel_mm_fsdp2_tp2`
(spmd_types,typechecking=True),放 b200 或 h100 都行;body 的 CI/CD 段随之改。

## 4. 保留、不动

- `_local_head_split()`:core 的 `local_qkv_head_split` 是 GQA forward 里的闭包、写死 `self.head_dim`,
  MLA 的 q 与 kv 头维不同,调不了。保留。它断言的 PartitionSpec 里也有 `"cp"`,
  要不要也标 TODO 由用户定。
- KDA 的 head-parallel 声明(`_set_kda_sharding`):所有参数在 tp mesh 上,kernel 在 `local_map` 后
  只算本地头。这是对的,和 core 问题无关。

## 5. 验证

- pyrefly 用 pre-commit 钉的版本跑一遍(本地没有),确认删掉的抑制不会让 CI 红。
- 数值:按 4500 口径,1/10/100 步 + 百分比 + 同提交控制行;10 步表已有,100 步与 tp4 行等
  `matrix_scripts/tp_a100/`。
- 提交信息无 trailer(现在 0),`git diff ac10ca48f <head>` 逐行读一遍再叫 ready(diff-audit 规则)。

## 6. body 要跟着改的

- Changed files:`distributed/utils.py` 撤掉或缩小;`kimi_k2_7/vision_encoder.py` 拆走后删行;
  `model.py` −1、`sharding.py` +4(已改)。
- Design 里 `clip_grad_norm_` 那条按第 1 节的结果重写或删除。
- CI/CD 段写上新格。
- 标题与 stack 段已改(main after 4527,4500 合并后 rebase)。

## 7. 同步

用户批准后 `git push --force-with-lease origin tp_sp_on_main:k3_tp_sp`;4499 的标题按 kit 第一行改。

---

# Appendix: the GPU box's draft of the same replies (TP/SP rework), written 2026-09-11 before the two lines were merged

Written independently on the GPU box (the reply drafts were committed there but never pushed, which is why they were missing here). Kept verbatim for the measured numbers, branch SHAs and test counts; where the two disagree the version above is the one to post.

# TP/SP rework handoff (2026-09-10), updated after the rebuild

The user's audit items and what the rebuilt branch does about each. Branch: `k3_tp_sp` = `tp_sp_on_main` (PR 4499), the rebuilt stack on main `ac10ca48f`; the pre-rebuild head `ce10692c1` and the reviewer's local commit `8a9c129d7` (pyrefly suppressions removed, four CP TODOs) are gone from the fork; `8a9c129d7` never reached origin and is not in this clone, so its two items were redone from scratch (below) and its TODO text has to come from the reviewer.

| item | result |
| --- | --- |
| 1. core `distributed/utils.py` (+58/-17): verify, likely revert | Verified first (`TP_SP_REWORK_2026-09-10.md` §1): with `utils.py` back to main, tp2 spmd_types is bitwise the branch (12.36934 / 10.17401 / 7.89447), tp2 partial_dtensor fails at the first clip on the mixed meshes. Reverted; the choice between declaring the tower on the tp mesh under partial_dtensor and refusing partial_dtensor + TP went to refusing (`7232c1210`): the tower's inputs and outputs would have to be wrapped as DTensors around a flex-attention forward, and spmd_types is the default backend. |
| 2. `kimi_k2_7/vision_encoder.py` (+3/-6): its own small PR | Its own commit at the bottom of the stack, `c0e1584df`, pushed as `k27_vision_tables_tp`; body `PR_BODY_K27_TABLES.md`. Evidence: K2.5 dp2 x tp2 with type checking fails on main at the tables with the retype error and gets past them with the commit (then hits K2.5's own qk-clip TP problem, reported separately). |
| 3. K3 TP CI cell | `kimi_k3_mm_tp2` in the b200 suite (`e6bed4bc9`): tp2, SP on, spmd_types, type checking, 2 GPUs. |
| 4. `_local_head_split`, KDA head-parallel declarations | Kept. |
| 5. Verification | Pinned pyrefly 0.45.1: 0 errors on the touched files. Numerics: the spmd_types cells bitwise before and after; the 10-step tables (this box, 4527-based twin) and the A100 100-step spmd_types rows in the body; partial_dtensor TP rows removed. `git diff ac10ca48f <head>`: 11 files, `utils.py` absent; grep audit clean. |
| 6. Body | Reworked (`PR_BODY_TP_SP_ON_4527.md`): no "both backends", the refusal and its reason in Summary and Limitations, tp=1 partial_dtensor control rows kept with a note, the 100-step A100 section, Tests with the CI cell and the K2.5 evidence, `utils.py` untouched. |
| 7. Sync | Done on the user's word: `k3_tp_sp` = `tp_sp_on_main`. |
| reviewer's `8a9c129d7` | The pyrefly suppressions: the one the branch itself adds (`preprocess_inputs`, `bad-override`) is not needed by the pinned checker and is dropped in the top commit; the other suppressions in the touched files are main's own. The four CP TODOs: text needed from the reviewer, not reconstructed. |

Head after the top commit: see `git log -1 origin/k3_tp_sp`.
