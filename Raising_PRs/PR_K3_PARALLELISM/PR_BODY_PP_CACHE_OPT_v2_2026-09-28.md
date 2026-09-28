# PR A body v2（只含第 3 类：接收缓冲按需分配、send 提早等掉），2026-09-28

## 状态（不粘贴）

- **用户 09-28：** "PR A只做第3类，然后同一个本地dev分支把其他的也留着"。自审和三方对比见 `PR_A_SELF_REVIEW_2026-09-28.md`。
- **分支：** review 分支 `pp_review_optimize` = `d37fb90f1`，main `f35966713` 上一个提交，3 个文件 +151/−3（`stage.py` +134，`__init__.py` +19，`test_kimi_k3_pp_stage.py` 在一个测试桩上补一行）。4312 的函数名、一跳的 [T, Nd, D] 格式、rank cache 和已审的测试都不动，#4914 的页面不受影响。不依赖 #4656。
  - 旧 head `439bd2088`（按 block 传输、bringer 释放加本提交的内容，叠在旧的列表提交 `aa6d9fedc` 上）备份为 `backup/pp_review_optimize_pre_20260928`，同时留在本地分支 `pp_review_optimize_dev`。
- **检查：** pyflakes、ufmt 干净；干净 worktree 上 Test plan 的四个文件 67 passed。新加的 3 行注释都是约束。
- **数字：** 5060 上（C 组布局，#4656 的列表提交之上）这一类把最重的 rank 从 11.47 降到 8.07 GiB（`PR_A_SELF_REVIEW_2026-09-28.md`）；main 之上的对比待测。body 里写 Pending (H100)。
- **标题建议：** `[Kimi K3] PP stage: receive buffers on demand, sends waited once the receiver has used them`
- **torch issue：** 草稿 `TORCH_ISSUE_PP_BUFFERS_2026-09-27.md`，开好后把号填进 Design 里的 `<torch issue link>`。

--- PR A body: PASTE BEGIN ---

## Summary

The Kimi K3 pipeline stage allocates each receive buffer when its receive is posted and waits its sends once the receiver has used them, instead of holding one receive buffer per micro-batch for every input and input gradient and waiting every send at the end of the step.

- `kimi_k3/pipeline_parallel/stage.py`: receive buffers are built from the tensor metadata when the receive is posted and dropped once read, for activations and input gradients alike. Under the action-list runtime, forward sends are waited at the stage's backward of the micro-batch, and input-gradient sends at the first later forward on the rank that consumes what the receiver produced after using them.
- `kimi_k3/pipeline_parallel/__init__.py`: turns the send waits on under `_PipelineScheduleRuntime` and derives the wait points from the schedule's `pipeline_order`.

## Design

torch's pipelining keeps a receive buffer per micro-batch for every input and input gradient for the whole step, and its action-list runtime waits every send at the end of the step while the pending work keeps the sent tensor alive. With interleaved schedules and many micro-batches these buffers and pinned sends hold a large share of a rank's memory at its peak. The stage overrides `_setup_forward_recv_info`, `_setup_backward_recv_info`, `get_fwd_recv_ops`, `get_bwd_recv_ops`, `get_fwd_send_ops` and `get_bwd_send_ops` to allocate late and wait early. Both behaviors hold for every pipeline model, so they belong in torch (<torch issue link>), and the overrides go once torch offers them.

The block transport, the rank cache and their tables are unchanged.

## Results

Pending (H100).

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py -q` (67 passed): the existing four-rank gloo pipelines under Interleaved1F1B and 1F1B, cache on and off, with every block gradient bitwise with one device, now run through the late receives and the early send waits.

--- PASTE END ---
