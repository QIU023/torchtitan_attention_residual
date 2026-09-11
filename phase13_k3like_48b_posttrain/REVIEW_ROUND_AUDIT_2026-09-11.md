# 三条线的评审轮次核验（2026-09-11）

对象：PP #4312（Tianyu 10 条）、TP/SP #4499（Shuhua 10 条）、QB #4412（被 #4577 接管）。
分支：`pp_review4`=`c6cea394e`、`k3_pp_transport`=`b815e03fc`、`tp_sp_on_main`=`d4d6e774c`、
`k27_vision_tables_tp`=`d8120354e`、`lora_review2`=`72bbcb639`。upstream/main=`116b4f7ae`。

## 1. PP #4312 — 6 个新提交 + 传输拆分，两个阻塞项

Tianyu 2026-09-11 01:58-02:12 的 10 条与修复的对应：

| 评论 | 位置 | 处理 |
|---|---|---|
| need docstring / unclear / 避免全局 env / 为什么建跨 rank 通信器 / 这个类同样的问题 | `pipeline_parallel.py:46,53,57,60,111` | 两个传输提交整体移出到 `k3_pp_transport`，PP 分支不再有 env 变量、`parallel_dims` 组和 `_NeighborP2P*` |
| 不信服需要这个 | `parallel_dims.py:566` | 同上，`parallel_dims.py` 从 diff 消失 |
| 能叫 `PPRankLocalCache` 吗 | `parallelize.py:269` | `9e6e5b9a8` 改名 |
| 这个在 GPU 上吧？ | `pipeline_stage.py:55` | `9e6e5b9a8` docstring 写明常驻 device、何时释放 |
| 这两个属性够做聪明事了 | `pipeline_stage.py:176` | `0f189b93c` 层→stage 映射直接从 split 读，删掉 all-gather 与本地遍历 |
| 数值差距不信服，举例说明累加顺序、怎么证明不是 bug | `parallelize.py:267` | **未回复**（要写文字，不是代码） |

另外两个提交是主动重构：`2b9b1a788` 把 K3 自己的 split 函数删掉，改为给 core 的
`_generate_llm_fqn_per_model_part` / `pipeline_llm` 加 `first_stage_modules` / `last_stage_modules`；
`b9e7cf8e7` 让 `module_fqns_per_model_part` 与 `pipeline_parallel_layers_per_stage` 互斥。

**阻塞 A：pp8 x vp4 recipe 丢了 vision encoder。** recipe 现在自己写死
`module_fqns_per_model_part`，只传了 `last_stage_modules`，没有 `first_stage_modules=("vision_encoder",)`；
而 `module_fqns_per_model_part` 一旦给出，`pipeline_llm` 就忽略 K3 入口传的 pinned 模块。本地复算
该 split：32 段、33 层齐全、末段 `norm/lm_head/output_res_proj/output_res_norm`，但
`vision_encoder` **不在任何一段里** → `_split_module` 把它在每段置 None。K3 debug flavor 是多模态的
（`_debugmodel` 里 `vision_encoder=_vision_encoder_config(...)`），dataloader 是 `cc12m-test`，
第一段拿到 `pixel_values` 后 `_prepare_multimodal_embeds` 直接
`raise ValueError("pixel_values were provided without a vision encoder.")`。
CI 格 `kimi_k3_pp8_vp4`（8 卡）必红。修法：recipe 的 `_generate_llm_fqn_per_model_part(...)`
补 `first_stage_modules=("vision_encoder",)`。

**阻塞 B：`2b9b1a788` 与上游刚合的 #4560 撞车。** 2026-09-10 合入的
`310e2a66e`（Refactor pipeline_vlm）把 `pipeline_vlm` 改成了公开入口
`pipeline_with_first_stage_modules(model, first_stage_module_fqns=[...])`，做的正是"把模块钉到第一段"，
而且是包一层 `pipeline_llm`、不改生成器签名。我们的 `first_stage_modules` 参数与它功能重叠
（顺序也不同：上游 `insert(0)` 前插，我们在 `tok_embeddings` 之后追加），rebase 必冲突，
reviewer 会直接问"为什么不用刚加的那个"。建议改成：first-stage 侧走 `pipeline_with_first_stage_modules`，
只把 **last-stage** 的需求作为增量提（给那个函数加 `last_stage_module_fqns`，或在生成器上只加末段参数）。
K3 需要末段（AttnRes 聚合跟着 head）是上游现在确实没有的能力，这是干净的卖点。

**阻塞 C：分支基还是老的。** `pp_review4` 的基仍是 `6e2ac3dcd`，落后 upstream/main 63 个提交，
#4312 在 GitHub 上从 #4527 合入起就是 `dirty`。我 09-10 做的 `pp_review5`=`6042863a4`
（同一条线 rebase 到 `d398a8fb9`，冲突两处已解）没有被用上，这 6 个新提交是在老基上做的。
顺序建议：先把 6 个新提交摘到 `pp_review5` 上（或重做一次 rebase），再补 A/B。

**次要：** recipe 里 `from torchtitan.distributed.pipeline_parallel import _generate_llm_fqn_per_model_part`
是跨文件引用私有函数——reviewer 之前就对同样的写法提过意见（"seems calling a private function across file"）。
改用 B 里的公开入口可以一并消掉。

**核对过没问题的：** 生成器加默认参数后 1065 个 (stages, layers, input_w, output_w) 形状与原函数逐字相同；
pinned 模块落位正确；`b5bb7b315` / `c6cea394e` 只是缓存标志与类型标注。

## 2. TP/SP #4499 — Shuhua 10 条，8 条已改

`tp_sp_on_main` 重建为 4 个提交，基 `da2f82670`（只落后 main 1 个）。

| 评论 | 处理 |
|---|---|
| `model.py:374` 这行移到 `set_kimi_k3_sharding_config` 之前 | 已改：`Decoder.Config.update_from_config` 先跑 |
| `model.py:428` 与其他 sharding 协议架构不一致 | 已改：改用 core 的 `multimodal_input_sharding()` |
| `model.py:473` 参考 qwen3.5 的 TP 做法 | 已改：MLA 的头拆分改用 core 的 `local_head_split`，本地副本 `_local_head_split` 删除 |
| `sharding.py:47` 先去掉 cp | 已改：`CP` 别名删除，model.py 的 4 处 cp 归零（`dense_activation_placement(cp=...)` 是 core 的必填参数，K2.7/qwen3.5 也都传 `cp=spmd.S(0)`，保留正确） |
| `sharding.py:292` 合并进 `set_kimi_k3_sharding_config` | 已改：对外只剩一个入口，带 `enable_tp` / `enable_sp` |
| `sharding.py:390` 复用 MoonViT 的 TP 路径，不要先设后覆盖 | 已改：塔直接用 K2.5 的 MoonViT 方案（colwise/rowwise linear、invariant norm/表） |
| `common/attention.py:116` 复用后可回退 | 已回退（文件从 diff 消失） |
| `common/linear.py:113` 同上 | 已回退 |
| `test_kimi_k3_sp_splice.py` 不需要单测 | 已删 |
| `kimi_k2_7/vision_encoder.py:123` K3 能不能直接复用 K2.7 的 vit TP | **保留 +3/−6**，但已拆成栈底独立提交 `ddb306332` 和独立分支 `k27_vision_tables_tp`（基 `ac10ca48f`，单文件）。**需要一句回复**：K3 确实复用了 K2.7 的方案，这个文件的改动不是为 K3 加功能，而是 K2.7 自己在 tp>1 + 类型检查下会报的 src 检查问题（CI 没有 K2.7 的 TP 格所以没人撞见），所以单独提。 |

其余核对：`TODO: rebase after upstream CP merge` 已随 cp 声明一起消失；K3 目录里剩的
`pyrefly: ignore` 三处都是 main 自己就有的（`model.py:410/412/457`）；新增 CI 格
`kimi_k3_mm_tp2`（b200，dp1×tp2，SP on，spmd_types + 类型检查，ngpu=2）。
body `PR_BODY_TP_SP_ON_4527.md` 还没跟上这次重建（头部 SHA、`partial_dtensor` 行、Changed files 全要重算）。

## 3. QB — #4412 被 #4577 接管

Shuhua 在 09-10 10:36（给我们留完 7 条评论之后）开了 **#4577 `[moe] add quantile-balanced moe routing`**，
`k3-qb`=`ebf6cc451`，基 `183efed45`，mergeable/clean。10:45 在里面 @QIU023 @JavaZeroo "can you help review here?"。
Jeremalloch（Marin）15:46 提出应该放到通用 MoE 组件里，Shuhua 21:58 已照做；他 09-11 03:48/03:49
又提了两条行内问题。**我们到现在一条都没回**（#4577 上没有任何 QIU023 的评论）。

#4577 与我们 #4412 的关系：算法同源（报告 2.3.3 + 附录 D），但落点不同——
它把 `QuantileBalancedTopKRouter` + `QuantileBalancer` 放进 `models/common/moe.py`，
hook 放进 `components/optimizer/optimizer.py`（`register_moe_quantile_balancing_hook`），
K3 只改 registry 的两行；并且顺手解决了 Shuhua 给我们提的 "top_k done twice"（一次 Top-(k+1)，
第 k+1 个分数当 cutoff）和 JavaZeroo 提的 eval 问题（`_select_experts` 里 `if not self.training` 走父类）。
我们的 #4412 实际上已被取代，应当去 #4577 做实质评审并把证据带过去。

本地核验（独立复算，非读码结论）：把 `observe` / `estimate_expert_bias` 抽出来跑
E=8、k=2、T=4096 的偏斜 logits，负载 max/ideal 由 3.43 收敛到 1.03（6 步），
`counts_E * k/E` 的分位反演方向正确，均值中心化把 `lower_bound` 消掉所以只传 bin 宽度也对。
直方图为空时 `counts_in_bin_E = 0` → `fraction_E = nan` → 写进持久 buffer，确认可复现。
`loss` mesh = dp_replicate × dp_shard × cp，不含 pp，所以 PP 下各 pp rank 层数不同不会撞 all-reduce（虚惊）。
`expert_bias_E` 是 `dense_param_placement(tp=spmd.R)`，EP 下不分片，所以 `min()/max()` 定出的
bin 范围各 rank 一致（另一个虚惊）。`observe` 在 `_select_experts` 里，而 core 用
`remat.region(..., recompute=False)` 包了它，所以全 AC 下也只计一次——sign rule 那个 `// 2` 的
hack 在这里不需要。

### 给 #4577 的评审稿

--- PASTE BEGIN (top-level comment) ---

Reviewed against the report (2.3.3 and Appendix D) and against our implementation in #4412; the shared placement is the right call, and the Top-(k+1) cutoff also removes the double top-k we were carrying. I re-derived the estimator standalone (E=8, k=2, T=4096, skewed logits): the load ratio goes from 3.43x to 1.03x of ideal in six updates, so the quantile inversion and the mean-centering are right. Two things I checked that are NOT problems, in case they come up: the `loss` mesh is dp x cp with no pp axis, so pipeline ranks holding different layer counts never meet in the stacked all-reduce; and `observe()` runs inside `_select_experts`, which core wraps in `remat.region(..., recompute=False)`, so the histogram is counted once per micro-batch even under full AC -- the `// 2` correction the sign-rule hook carries is not needed here.

Two comments inline. I can also run our 10-step dp1 / dp2 x ep2 / dp8 x ep8 matrix on this branch against the sign rule on the parent commit (same seed, same batch, one inductor cache) and post it here; on our branch step 1 was bitwise with the sign rule and the grad norm moved 25% by step 3, which is the kind of number that gets questioned, so it is worth having on the PR that lands. Happy to close #4412 in favour of this.

--- PASTE END ---

--- PASTE BEGIN (inline: moe.py, estimate_expert_bias, the fraction_E line) ---

`counts_in_bin_E` is zero when the histogram is empty, so `fraction_E` is NaN and the `copy_` below writes NaN into `expert_bias_E`, which is persistent and never recovers. The default trainer always has a training forward behind every `step()`, but a step that only ran validation micro-batches, or a custom loop, reaches it. One guard is enough: return `local_expert_bias_E - local_expert_bias_E.mean()` when `counts_E.sum() == 0`.

--- PASTE END ---

--- PASTE BEGIN (inline: moe.py, the histogram buffer / num_bins) ---

At the released K3 shape this is not small: `(E=384, B=1000)` int32 is 1.5 MB resident per MoE layer, and `torch.stack(histograms)` in the hook materializes every layer at once -- about 90 MB, plus the all-reduce buffer, once per step. The per-micro-batch temporaries in `observe` scale the same way (`bin_indices_TE` is int64 `(T, E)`). Worth either reducing layer by layer at a chosen chunk size, or documenting `num_bins` as the memory knob it is.

--- PASTE END ---

## 4. LoRA

`lora_review2` 多了一个 `72bbcb639`（删掉 packed TP matmul 不再用的 `Replicate` import），
纯清理。`PR_BODY_LORA.md` 头部还写着 `aaa823064`，要改成 `72bbcb639`。
