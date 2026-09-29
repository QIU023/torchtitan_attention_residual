# PR A body v3（回到 AttnRes 论文的 cache 下界），2026-09-29

## 状态（不粘贴）

- **用户 09-29：** "这个PR A的目的是不是drift了？？？我们原始的原因是实现attn res论文的cache理论下界"，随后确认新范围（"4个都可以"）。
- **分支：** review 分支 `pp_review_optimize` = `4ddf8e912`，#4656 `5d469fdf3` 之上一个提交，6 个文件 +493/−274。旧 head `85eefa54b`（只有第 3 类）备份在 `backup/pp_review_optimize_pre_20260929`。
- **内容：** 第 1 类（一个 block 一个张量，store 存引用，stage 不再拼 stack，模型对外收发列表）、第 2 类（在带它进来的 stage 的反向释放）、send 早等（torch 还没有，stage 里覆盖）。不再覆盖接收缓冲：torch#196463（09-22）已经在收之前才分配，原来的覆盖在 09-23 以后的 nightly 上会报 `PipeliningMetadataError`。本地 dev 分支 `1777ad806` 的内容全部并入。
- **依赖：** 叠在 #4656 上（模型内部的 block 列表）。#4765 = `019462171`、#4764 = `2a719e512`、`c4afb61f4` 重叠到本 PR 上。
- **检查（`kit_overnight_2026-09-29/`）：** torch 2.15.0.dev20260928+cu130 上 Test plan 的五个文件 71 passed；pyflakes 干净；ufmt 按仓库钉的版本（2.3.0 / black 22.12.0 / usort 1.0.5）干净。逐行读过 diff：删掉接收覆盖后没有留下死代码；新增注释 12 行都是约束，`_collect_into` 的两行注释压成一行。提交信息没有 trailer，没有跨仓库引用。
- **数字：** 5060 上 #4656 对本 PR 的五个布局在 `OVERNIGHT_RESULTS_2026-09-29.md` T1d（只进 logbook）。body 里写 Pending (H100)。
- **标题建议：** `[Kimi K3] PP: store each attention-residual block once per rank and free it at its last reader`
- **torch issue：** 草稿 `TORCH_ISSUE_PP_BUFFERS_2026-09-27.md` 已改成只提 send；开好后把号填进 Design 的 `<torch issue link>`。
- 粘贴区查过 we/our/us 和破折号。

--- PR A body: PASTE BEGIN ---

## Summary

The Kimi K3 pipeline stores each attention-residual block once per pipeline rank and frees it at its last reader, the memory the Attention Residuals paper states for cross-stage caching ("each block is stored exactly once across all V virtual stages", section 4.1).

- `kimi_k3/pipeline_parallel/stage.py`: each hop sends one tensor per block, the rank store keeps references to the received and committed tensors, and a stage hands its blocks to the model as a list instead of stacking them into a fresh leaf. The store drops each block at the backward of the stage that brought it onto the rank. Under the action-list runtime, a forward send is waited at the stage's own backward of that micro-batch, and an input-gradient send at the first later forward that proves the peer has used it.
- `kimi_k3/pipeline_parallel/cache.py`: release by block, and gradient deposits that add in place.
- `kimi_k3/pipeline_parallel/__init__.py`: turns the send waits on under `_PipelineScheduleRuntime`, with the wait points taken from the schedule's `pipeline_order`.
- `kimi_k3/model.py`: the model takes and returns its blocks as a list.

## Design

The cached transport of #4312 sends each block once, but a rank holds it several times. Every stage stacks the blocks it reads into a fresh `[T, N, D]` leaf on entry and keeps that leaf until its backward, and the store keeps views that pin whole received payloads and the model's output stack. A block read by k stages on a rank therefore exists 1 + k times. With one tensor per block, the received tensor is the block and a committed block is the model's own tensor; the store and the stages hold references to them, so each block exists once on the rank.

In backward order, the last reader of a block on a rank is the stage that brought it there, by receiving it or by committing it. The store drops the block at that stage's backward, which is earlier than the end of the micro-batch that the paper's accounting assumes.

A pending send keeps its tensor alive until it is waited, and torch's action-list runtime waits every send at the end of the step, which would keep a released block alive until then. The stage waits a forward send at its own backward of the micro-batch, when the receiver has used it, and an input-gradient send at the first forward on the rank that consumes what the peer produced after using the gradient; a send without such a point keeps the end of step wait. This behavior belongs in torch (<torch issue link>), and the override goes once torch has it. Single-stage schedules fuse each send with a receive, so the early waits apply only under the action-list runtime. Receive buffers are left to torch, which allocates each one right before its receive.

## Results

Pending (H100).

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py tests/unit_tests/cpu/test_kimi_k3_attention_residual_recompute.py -q` (71 passed): four-rank gloo pipelines under Interleaved1F1B and 1F1B, blocks opening inside stages, every block gradient bitwise with one device, each block released by the stage that brought it, and the gradient wait points checked against `pipeline_order`.

## Relation to other PRs

- Stacks on #4656, which carries the blocks as a list inside the model.
- #4765 and #4764 stack on this PR.

--- PASTE END ---
