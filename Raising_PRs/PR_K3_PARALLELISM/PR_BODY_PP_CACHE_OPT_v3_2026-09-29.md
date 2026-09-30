# PR A body v3（回到 AttnRes 论文的 cache 下界），2026-09-29

## 状态（不粘贴）

- **用户 09-29：** "这个PR A的目的是不是drift了？？？我们原始的原因是实现attn res论文的cache理论下界"，随后确认新范围（"4个都可以"）。
- **09-29 夜 raise 前检查（用户："改"）：** review 分支 `pp_review_optimize` = `5a8163a58`（旧 head `ec8bb420a` 在 `backup/pp_review_optimize_pre_20260929c`），底下还是 rebase 后的 #4656 两个提交。
  - 改动只为类型检查：`stage.py` 的 `fwd_cache` 直接用 `input_tensors + list(flatten_args(...))`，不再有 `list[torch.Tensor]` 标注那行；反向按 block 逐个取梯度，拼 `grad_delta` 时显式建列表。行为不变。提交信息加了一句 `PP_ATTN_RES_CACHE.md` 和图例的改动。
  - pyrefly 0.45.1（只读跑）：main `d6810e4d6` 16 个错误（15 个是本机缺 `torch_checkpointing`、`rich` 的 import，1 个是 main 自己的 `int * None`）；PR A 修前 18、修后 16，和 main 一样。分范围 pre-commit（跳过 pyrefly、no-commit-to-branch）全部通过，没有改文件。71 passed。
  - CI 格（B200 那套里 K3 的三格，5060，torch 2.15.0.dev20260928，recipe 原样）：PP 格 `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4`（8 卡）在 PR A `ec8bb420a`、`5a8163a58` 和 rebase 后的 #4656 `4ae9422db` 上都 rc=0，10 步；`mm_muon`（2 卡）两边 rc=0；`mm`（4 卡）两边都 rc=1，`SPMD type mismatch on axis mesh_ep: P vs I`，main `d6810e4d6` 本身在同一环境同样失败，不是这条线引入的。
  - 这套 recipe 不设 seed、每格自己的 cache，step 1 的 loss 每次都不一样（8.17270 / 8.23765 / 8.21091），所以这些格只看 rc，不比 loss；逐位对比仍是 pp4 × vpp2 那组（同 seed、同一份 cache，20/20）。记录在 `kit_overnight_2026-09-29/ci_cells/`。
  - 粘贴区只改 Relation：说明前两个提交是 rebase 后的 #4656。
  - raise 前还差：Design 里的 `<torch issue link>`（issue 开了填号，或者删掉括号那半句）；Results 等 H100。
- **09-29 晚 rebase（用户："b"，"但是后续PR A整条线是建立在4656 rebase main之后"）：** review 分支 `pp_review_optimize` = `ec8bb420a`，在 upstream main `d6810e4d6`（含 #4914）上：底下两个是 rebase 后的 #4656（`9c6904dea`、`4ae9422db`，diff 和 `2516926f3`、`5d469fdf3` 一样，只解了 #4905 改名那处 import 冲突），第三个是本 PR。#4656 自己的 PR 分支 `k3_ac_reuse_attention` 和 review 分支 `attnres_review1` 没动，还是 `5d469fdf3`；review 要它 rebase 时，直接用这两个提交。旧 head `4ddf8e912` 在 `backup/pp_review_optimize_pre_20260929b`。
  - 本 PR 这次一起改了 #4914 加的页面：`PP_ATTN_RES_CACHE.md` 的表（一跳带什么、rank 留什么、什么时候放）、1F1B 那段，加一段 send 早等；stage 那张图只改图例里用"stack / 列"说的几行（B、Δ、⊕、held、split、deposit、leaf，另加 cache 关那一栏的副标题），框和箭头不动。本 PR 删掉了 `_pack_outgoing_delta`，#4914 改过它的 docstring，这是 rebase 时唯一的冲突，取本 PR 这边。
  - 8 个文件 +517/−292。新 torch 上 71 passed，pyflakes、ufmt（钉的版本）干净。5060 上 pp4 × vpp2 复核：20/20 逐位相同，loss、各 rank 峰值、block 记账和换底前一个数不差。
- **分支（09-29 下午）：** review 分支 `pp_review_optimize` = `4ddf8e912`，#4656 `5d469fdf3` 之上一个提交，6 个文件 +493/−274。旧 head `85eefa54b`（只有第 3 类）备份在 `backup/pp_review_optimize_pre_20260929`。
- **内容：** 第 1 类（一个 block 一个张量，store 存引用，stage 不再拼 stack，模型对外收发列表）、第 2 类（在带它进来的 stage 的反向释放）、send 早等（torch 还没有，stage 里覆盖）。不再覆盖接收缓冲：torch#196463（09-22）已经在收之前才分配，原来的覆盖在 09-23 以后的 nightly 上会报 `PipeliningMetadataError`。本地 dev 分支 `1777ad806` 的内容全部并入。
- **依赖：** 叠在 #4656 上（模型内部的 block 列表）。#4765 = `6c9943a91`、#4764 = `d248662a8`、`43ccdcc06` 叠在本 PR `5a8163a58` 上（09-29 夜；之前是 `019462171`、`2a719e512`、`c4afb61f4`）。
- **检查（`kit_overnight_2026-09-29/`）：** torch 2.15.0.dev20260928+cu130 上 Test plan 的五个文件 71 passed；pyflakes 干净；ufmt 按仓库钉的版本（2.3.0 / black 22.12.0 / usort 1.0.5）干净。逐行读过 diff：删掉接收覆盖后没有留下死代码；新增注释 12 行都是约束，`_collect_into` 的两行注释压成一行。提交信息没有 trailer，没有跨仓库引用。
- **数字：** 5060 上 #4656 对本 PR 的五个布局在 `OVERNIGHT_RESULTS_2026-09-29.md` T1d（只进 logbook）。body 里写 Pending (H100)。
- **标题建议：** `[Kimi K3] PP: store each attention-residual block once per rank and free it at its last reader`
- **torch issue：** 草稿 `TORCH_ISSUE_PP_BUFFERS_2026-09-27.md` 已改成只提 send；开好后把号填进 Design 的 `<torch issue link>`。
- 粘贴区查过 we/our/us 和破折号。
- **09-30 精简（CPU 这边，用户："PR A的body文件有什么能简化的吗？"）：** 只动 Summary 和 Design。
  - Summary 里 stage.py 那条原来三句，和 Design 几乎逐句重复，缩成一行；文档那条也缩短了。
  - Design 删了 "exists 1 + k times"：#4656 的模型在 PP 边界还会把所有块 stack 成一次输出，store 里 committed 块的 view 会钉住整个输出 stack，所以份数不止 1 + k，现在只写 "several times"。
  - Design 第三段删了三处：没有等待点的 send 照旧步末等（细节）；单 stage schedule 把 send 和 receive 合批、所以只在 action-list runtime 下早等（`__init__.py` 的注释和 Summary 已经说了）；接收缓冲交给 torch（这是相对旧版本 PR A 说的，本分支本来就不碰）。
  - `<torch issue link>` 还在，raise 前填号或删掉括号。

--- PR A body: PASTE BEGIN ---

## Summary

The Kimi K3 pipeline stores each attention-residual block once per pipeline rank and frees it at its last reader, the memory the Attention Residuals paper states for cross-stage caching ("each block is stored exactly once across all V virtual stages", section 4.1).

- `kimi_k3/pipeline_parallel/stage.py`: one tensor per block on each hop and in the rank store, each block released at the backward of the stage that brought it, and early send waits under the action-list runtime.
- `kimi_k3/pipeline_parallel/cache.py`: release by block, and gradient deposits that add in place.
- `kimi_k3/pipeline_parallel/__init__.py`: turns the send waits on under `_PipelineScheduleRuntime`, with the wait points taken from the schedule's `pipeline_order`.
- `kimi_k3/model.py`: the model takes and returns its blocks as a list.
- `kimi_k3/pipeline_parallel/PP_ATTN_RES_CACHE.md` and the stage figure's legend: one tensor per block, the new release point and the send waits.

## Design

With the cached transport of #4312, a rank holds a block several times: every stage stacks the blocks it reads into a fresh `[T, N, D]` leaf that lives until its backward, and the store's views pin whole received payloads and output stacks. Here a block is one tensor, the received one or, for a committed block, the model's own, and the store and the stages hold references to it, so it exists once on the rank.

In backward order, the last reader of a block on a rank is the stage that brought it there, by receiving or committing it, so the store drops it at that stage's backward, earlier than the end of the micro-batch that the paper's accounting assumes.

A pending send keeps its tensor alive until it is waited, and torch's action-list runtime waits every send at the end of the step, which would keep a released block alive until then. The stage waits a forward send at its own backward of that micro-batch, and an input-gradient send at the first forward on the rank that consumes what the peer produced after using it. This belongs in torch (<torch issue link>); the override goes once torch has it.

## Results

Pending (H100).

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py tests/unit_tests/cpu/test_kimi_k3_attention_residual_recompute.py -q` (71 passed): four-rank gloo pipelines under Interleaved1F1B and 1F1B, blocks opening inside stages, every block gradient bitwise with one device, each block released by the stage that brought it, and the gradient wait points checked against `pipeline_order`.

## Relation to other PRs

- Stacks on #4656, which carries the blocks as a list inside the model. The first two commits here are #4656 rebased onto current main; they show here until #4656 merges.
- #4765 and #4764 stack on this PR.

--- PASTE END ---
