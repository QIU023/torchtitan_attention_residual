# PP 4312 待修清单 — 给 GPU 盒子（2026-09-11，CPU 盒子核对 round 3 后）

对象 `pp_review4` = `c6cea394e`（6 个 round-3 提交在老基 `6e2ac3dcd` 上）。
下面三条按优先级，都在代码里，改完再发 `Raising_PRs/PR_K3_PARALLELISM/REPLY_4312_2026-09-11.md`
里对应的回复。第 4 节是核对时算出来的数据，留档给 body 和 comment 1 的回复用。

## 1. pp8 x vp4 recipe 丢了 `vision_encoder`（CI 必红）

`torchtitan_recipes/tests/features.py::kimi_k3_debugmodel_pp8_vp4` 现在自己写死
`module_fqns_per_model_part`，只传了 `last_stage_modules=("output_res_proj", "output_res_norm")`。
而 `pipeline_llm` 只在 `module_fqns_per_model_part is None` 时才应用 K3 入口传的 pinned 模块
（`2b9b1a788` 加的新参数就在那个分支里）。于是 `vision_encoder` 不出现在任何一段的 FQN 列表里，
`_split_module` 把它在 32 段上全部置 None。

K3 debug flavor 是多模态的（`_debugmodel` 里 `vision_encoder=_vision_encoder_config(...)`），
recipe 用的 dataloader 是 `cc12m-test`，第一段拿到 `pixel_values` 后
`_prepare_multimodal_embeds` 直接
`raise ValueError("pixel_values were provided without a vision encoder.")`。

本机用 core 的生成器复算该 split 确认：32 段、33 层齐全、末段 `norm/lm_head/output_res_proj/
output_res_norm`，`vision_encoder` 不在其中。

改法：recipe 的 `_generate_llm_fqn_per_model_part(...)` 调用补
`first_stage_modules=("vision_encoder",)`。跑一次 `kimi_k3_pp8_vp4` 格确认。

## 2. `b9e7cf8e7` 的 post_init 检查打断 core 自己的 split 注入

`dataclasses.replace()` 会重新走 `__init__` 因而重新触发 `__post_init__`（本机验证）。core 注入
生成的 split 用的正是这个写法：

```python
if parallelism.module_fqns_per_model_part is None:
    fqn_per_part = _generate_llm_fqn_per_model_part(num_virtual_stages, ...)
    parallelism = dataclasses.replace(parallelism, module_fqns_per_model_part=fqn_per_part)
```

- 老基上：`pipeline_parallel.py:196`（`pipeline_vlm`）和 `muse_glimmer/parallelize.py:201`。
- 今天的 main 上：#4560 统一成 `pipeline_with_first_stage_modules`，`kimi_k2_7`、`muse_glimmer`、
  `qwen3_5`、`qwen3_6`、`qwen3_8` 五个模型都走它。

所以这五个模型只要用户设了 `--parallelism.pipeline_parallel_layers_per_stage=N`，就在 replace
那行抛 `ValueError: ... both describe the pipeline split`；而那个 N 正是
`_get_pipeline_metadata` 用来算 `num_virtual_stages` 的合法输入。

CI 抓不到：全树只有 `llama3_debugmodel_fsdp2_pp2_1f1b_layers_per_stage` 和
`llama3_debugmodel_pp4_interleaved_1f1b_layers_per_stage` 设这个旋钮，llama3 直接走
`pipeline_llm` 不经过 replace；`tests/unit_tests/cpu/test_pipeline_parallel.py` 也没有把
`pipeline_with_first_stage_modules` 和 `layers_per_stage` 组合的用例；新加的
`test_pipeline_split_is_given_one_way` 只测直接构造。静默回归。

改法（保留 Tianyu 点名的 post_init 位置）：注入 split 的同时把已被消费的旋钮清掉，每个站点一行

```python
parallelism = dataclasses.replace(
    parallelism,
    module_fqns_per_model_part=fqn_per_part,
    pipeline_parallel_layers_per_stage=None,
)
```

语义上成立：FQN 一给出，`pipeline_llm` 就不再读 `layers_per_stage`（该分支上算出的
`num_virtual_stages` 无人使用）。补一个覆盖 `pipeline_with_first_stage_modules` +
`layers_per_stage` 的单测。备选是把检查挪到 `ConfigManager` 的用户输入校验，但不是他点的位置。

回复稿见 `REPLY_4312_2026-09-11.md` 第 6 节末尾的 PASTE 块（把这个交互写成我们自己发现并修的）。

## 3. `2b9b1a788` 与 #4560 撞车，以及基太老

`310e2a66e`（#4560，09-10 合入）把 L144 的 `pipeline_vlm` 改成了公开入口
`pipeline_with_first_stage_modules(model, first_stage_module_fqns=[...])`，做的正是"把模块钉到
第一段"，而且是包一层 `pipeline_llm`、不改生成器签名。我们现在的做法是给
`_generate_llm_fqn_per_model_part` / `pipeline_llm` 加 `first_stage_modules` 参数，与它功能重复
（插入顺序也不同：上游 `insert(0)` 前插，我们在 `tok_embeddings` 之后追加），rebase 必冲突。

Tianyu 当初点的就是 L144（评论 3984970518），顺着 #4560 改反而更好卖：
first-stage 侧直接用 `pipeline_with_first_stage_modules`，**只把 last-stage 作为增量提**
（给那个函数加 `last_stage_module_fqns`）。AttnRes 聚合要跟着 head 是上游确实没有的能力。

同时 `pp_review4` 的基还是 `6e2ac3dcd`（落后 main 60+）。顺序建议：先把 6 个 round-3 提交挪到
`pp_review5`（`6042863a4`，已 rebase 到 `d398a8fb9`，两处 #4527 冲突已解）上或重做 rebase，
再改 1/2/3。

## 4. 93 层的块流量（核对 comment 1 时用 PR 自己的 `layout.py` 算的，留档）

`layout.py` 是纯 Python，可脱离 torch 直接驱动：用 core 的生成器出 split，
`layer_to_stage_from_split` 出映射，`stage_to_rank = {s: s % P}`，两种 `cache` 各跑一次。

93 层、块 12 → **8 个块**（7 个满 12 + 尾部 9），不是 9 个。单位：一个块 = 一个 `[T, D]`，每 micro-batch。

| 形状 | cached 总量 / 单跳峰值 | naive 总量 / 单跳峰值 | 比 |
|---|---|---|---|
| pp8 x vp1 (S=8) | 28 / 7 | 28 / 7 | **1.0** |
| pp4 x vp2 (S=8) | 18 / 3 | 28 / 7 | 1.6 |
| pp2 x vp4 (S=8) | **7 / 1** | 28 / 7 | **4.0** |
| pp8 x vp4 (S=32) | 52 / 2 | 136 / 8 | 2.6 |
| pp16 x vp2 (S=32) | 96 / 4 | 136 / 8 | 1.4 |
| pp4 x vp8 (S=32) | 24 / 1 | 136 / 8 | 5.7 |

S=8 时 naive 的逐跳载荷就是 `[1,2,3,4,5,6,7]`；pp2 x vp4 的 cached 是 `[1,1,1,1,1,1,1]`。

- naive ≈ B·S/2（每个块要在它提交之后的**每一跳**上再过一次线；只和总 stage 数 S 有关，与 rank 数无关）。
- cached ≤ (P−1)·B（收方在 stage s+1 上一次跑的是 s+1−P，已持有那之前提交的块；**与深度和 V 无关**）。
- 比值 ≈ S / (2(P−1)) = P·V / (2(P−1))：P 大时约 V/2，P=2 时正好等于 V。
- **V=1 时两者完全相同**（pp8 x vp1 实测都是 28）：每个 rank 只见每个 stage 一次，没有任何东西可缓存。
  这也是"pp2 上 cache on/off 十步逐位相同"的同一个原因。

"每个全局 stage 恰好持有一个块"⇒ S = 8（pp8 x vp1 / pp4 x vp2 / pp2 x vp4）。但 core 的生成器
给不出整块对齐的切分：93 层 8 段，默认权重是 `[11,12,12,12,12,12,12,10]`，把
`first/last_stage_less_layers` 设成 0 也只是 `[12,12,12,12,12,11,11,11]`，都不是 12x7+9；要真正
一块一段必须显式给 `module_fqns_per_model_part`。不过"块 b 在 stage b 上开"仍然成立（层 12b
落在 stage b），路由表意义上就是一段一块。

**待改**：`REPLY_4312_2026-09-11.md` 第 1 节（option 1 vs 2）里"It costs `[PP degree]` times the
payload on every hop"是照抄 Tianyu 的说法，不准确——倍数是 ≈ S/(2(P−1)) 且只在 V>1 时存在。
用户还没定要不要换成上表的实测（pp8 x vp4：每跳 2 块 vs 8 块，总量 52 vs 136）。

## 5. 其他仍开着的

- debug 深度 33 / 24 / 17 的取舍：回复稿第 5 节（用户方案：共享模型还原成 main 的 24 层跑
  pp2 x vp2，33 层只留给 pp8 x vp4）。上游 main 的共享 debugmodel 自 #4025 起一直是 **24 层**；
  我们这条线走过 24 →（24 + `debugmodel_32l`）→ 30 → 33。13 层只是我们自己的探针别名。
- 第 3 节（数值）以 `REPLY_4312_NUMERICS_2026-09-11.md` 为准，我写的那版作废（它把 cache on/off
  说成不同的累加顺序，与实测的逐位相同矛盾）。
- recipe 里跨文件 import 私有的 `_generate_llm_fqn_per_model_part`（rounds 汇总里记了正在删）。

## 6. #5/#6（层→stage 映射去掉 all-gather）与 #4560 的关系

问题：`0f189b93c` 去掉 all-gather 之后，`pipeline_kimi_k3` 需要知道"core 到底用了哪个 split"，
而 `pipeline_llm` 在内部算完 split 并不交还（只返回 schedule / model_parts / has_first / has_last）。
于是分支现在的做法是**在 `pipeline_llm` 返回之后把同一个 split 再算一遍**：
`_module_fqns_per_model_part()` 调用 core 的 `_get_pipeline_metadata()` + 
`_generate_llm_fqn_per_model_part()`，参数与 core 内部那次一致。

**#4560 本身不解决这个**——它没有碰层映射。但**它规定的形状可以让问题消失**：
`pipeline_with_first_stage_modules` 是**调用方**先算出 `fqn_per_part`，用
`dataclasses.replace` 塞进 `module_fqns_per_model_part`，再调 `pipeline_llm`。在这个形状下
K3 手里本来就握着刚算出的 split，`layer_to_stage_from_split(fqns)` 直接可用：一次推导、
没有私有 import、没有集合通信。这正是 `2b9b1a788` 之前 K3 的做法
（`kimi_k3_module_fqns_per_model_part` + replace），也正是 Tianyu "extend L144" 指的方向。

今天不是活 bug（已核：`_split_module` 先 `copy.deepcopy(whole_model)`，所以 `pipeline_llm` 返回后
原 model 完整，`_kimi_k3_first_stage_modules(model)` 在每个 rank 上一致；两边调的是同一对函数、
同样的入参，不会各算各的）。代价是两条：一个模型文件里硬编码了 core 的**两个私有函数及其调用
顺序**（round 1 的 3922481936 就是对这种写法提的意见），以及默默依赖 `pipeline_llm` 今后仍用同样
的方式推导 split。

结论：#5/#6 的干净解法与第 3 节是**同一个改动**——改走 #4560 的形状（给
`pipeline_with_first_stage_modules` 加 `last_stage_module_fqns`，K3 用它算一次 split、注入、
同一个对象喂给 `layer_to_stage_from_split`），#4/L144 和 #5/#6 两条评论一起结清。
注意这需要先做第 2 节的修复：K3 一旦恢复用 `dataclasses.replace` 注入 split，就会撞上
`b9e7cf8e7` 的 post_init 检查。三件事是一个连贯的改动，按 2 → 3 → 1 的顺序做。
