# PR A body v4（最小 diff 版，保留 4312 的结构），2026-09-30

## 状态（不粘贴）

- **09-30 H100 结果已填进粘贴区（`PRA_H100_2026-09-30.md`）：** Results 放了 dim 6144 的两个 4 卡布局（100 步逐位一致，每卡省 1.3 到 3.8 GiB，步时快约 2%，块占用落在论文界上）。dp2 × pp2 × vpp2、pp2 × vpp2、pp2 × vpp4 在 6144 下两个树都 OOM，dim 5120 在补跑，出来后再补一行。Design 改了一句：原文"块在带进来的 stage 反向后就释放"，实测块占用在论文界上、没到紧界，差的是前向 send 要到发送方自己反向时才等，所以改成"带进来的 stage 反向完、并且把它继续发出去的前向 send 也等完之后释放"。要不要这样写你定。
- **09-30 H100（115.124.123.240，`345e9e00a`，`PRA_H100_2026-09-30.md`）：** Test plan 五个文件在 H100 机器上 70 passed（基线 `e66a9442b` 也是 70 passed，含要 CuTeDSL 的 recompute 测试），粘贴区的 `<N>` 已填。dim 6144 的显存实验在跑，Results 等它。
- **用户 09-30：** "PR A怎么对cache和stage有这么多的改动？？？能尝试最小化diff吗？而且4312已经合并并且结构是清晰的，这个diff这么改没办法评审，重新重构简化"。
- **分支（用户 09-30："理想情况来说，不要直接覆盖pp review optimize，这个改动直接加一个commit到这个分支，我去测，测完了没问题再force with lease+rebase 4765/4764"）：** `pp_review_optimize` = `345e9e00a`，fast-forward 推送，没有 force。
  - 推之前远端已经被别的会话 rebase 到 upstream main `97e673b77`：`07abef619`、`e66a9442b`（#4656）、`36cfddf87`（v3），三个补丁和 `9c6904dea`、`4ae9422db`、`5a8163a58` 逐字相同（range-diff 全是 `=`）。
  - `345e9e00a` 加在 `36cfddf87` 上面，把 v3 缩成最小版（7 个文件 +252/−461）。它相对 #4656 的改动和本地旧底上的 `8b0fcbe38` patch-id 相同，所以 `8b0fcbe38` 和 kit 里的补丁作废，以分支为准。
  - 新底上重跑过：本机测试 17 passed（另 1 个是本机缺 `renderers` 的固有失败），CPU 记账四个 rank 都正好等于紧界，分范围 pre-commit 全过、没有改文件，pyrefly 0 个错误。
- **你测完以后：**
  1. 把 `36cfddf87` 和 `345e9e00a` 压成一个提交（树不变，提交信息用 `8b0fcbe38` 那条），force-with-lease 推 `pp_review_optimize`，旧 head 备份到 `backup/pp_review_optimize_pre_20260930`。
  2. #4765、#4764 rebase 到新 PR A 上：`stage.py`、`cache.py` 一定冲突，而且"在带进块的 stage 反向时释放"这个时点得由 #4765 自己带上（它在 cache 的 `put` 里 pin、按块 `release` 里 unpin）。
  3. Test plan 五个文件的通过数，在 GPU 机器上跑完填上。
- **和 v3（`5a8163a58`）对比：**

  | | v3 `5a8163a58` | v4 `8b0fcbe38` |
  |---|---|---|
  | 规模 | 8 个文件 +521/−292 | 7 个文件 +160/−140 |
  | `stage.py` | +254/−166；删掉 4312 的 `_assemble_stack`、`_pack_outgoing_delta`、`_split_stack_grad`，新写 `_outgoing_blocks`、`_route_input_grads`、`_brought`、`_grad_send_wait_points`、`_GradSendWaits` | +116/−95；4312 的函数名、`_order`/`_delta_in` 记账和流程都不变，只把"stack 的一列"换成"一个块一个张量" |
  | `cache.py` | +28（按块释放、deposit 原地加、`blocks()` 返回拷贝） | 不动 |
  | 释放点 | 改成带进块的 stage 的反向 | 不动（rank 最后一个 stage 的前向之后） |
  | 梯度 send 早等 | 有（按 `pipeline_order` 推等待点） | 没有 |
  | 前向 send 早等 | 有 | 有（本 stage 反向这个 micro-batch 时等） |
  | 测试 | 删改 4312 的测试，新增"按块释放""等待点"两个用例，+202/−89 | 4312 的测试原地改接口（测试名全部保留），不新增用例，+27/−28 |

- **删掉的东西为什么不影响下界：**
  - 释放点：按块传以后，store 里存的就是收到的张量和模型自己的张量，各 stage 的输入是它们的别名。这块内存本来就被 autograd 和前向缓存持有，一直到带它进来的 stage 做完反向。所以 store 提早放掉引用不会让内存提早释放，也不会拖晚；4312 原来的释放点可以不动。
  - 梯度 send：GPU 那边 T1d 的探针算 block 占用时只算前向 send，梯度 send 单列（`bwd_sends_alive_gib`）。梯度 send 扣住的是梯度，不是块，和论文的 cache 下界无关。它仍然照 torch 默认在步末等；要优化可以另做，或者等 torch issue。
- **本机验证：**
  - 测试（harness，torch 2.13 CPU）：`test_kimi_k3_pp_stage.py`、`test_kimi_k3_pp_block_grads.py`、`test_kimi_k3_pp_layout.py` 17 passed。另外 1 个是本机固有的失败（缺 `renderers` 模块），和改动无关，改动前的基线也是这样。逐位流水线测试里前向 send 早等是打开的。
  - CPU 记账（`kit_pra_min_2026-09-30/cpu_bound.py`，4 个 gloo rank，pp4 × vpp2，Interleaved1F1B，8 个 micro-batch，玩具模型每个块是独立张量；每个动作之后，把 store、各 stage 前向缓存里的块输入输出、还没等的前向 send 按存储去重求和，和紧界比）：

    | 树 | 各 rank 峰值（块数） | 紧界 | 最多超出紧界 |
    |---|---|---|---|
    | 基线（#4656 `4ae9422db`，stage 是 4312 的） | 48 / 65 / 76 / 67 | 23 / 26 / 24 / 20 | 32 到 52 |
    | v3 `5a8163a58` | 23 / 26 / 24 / 20 | 同上 | 0 |
    | v4 `8b0fcbe38` | 23 / 26 / 24 / 20 | 同上 | 0 |

  - 检查：分范围 pre-commit（ufmt 钉的版本、flake8、pydoclint、codespell、trailing-whitespace、end-of-file、check-ast、insert-license）全部通过；pyrefly 0.45.1 只读跑三个源文件，0 个错误，和基线一样。新增注释三行，都是一行的约束：action-list runtime 不把 send 和 receive 合批；接收方用完张量的时点；core 只保留收到的输入的梯度。提交信息没有 trailer，没有跨仓引用。
- **还缺：** Test plan 的通过数（五个文件要在 GPU 机器上跑；本机缺 CuTeDSL，recompute 测试跑不了），Design 里的 `<torch issue link>`，Results 等 H100。
- **标题建议：** `[Kimi K3] PP: keep each attention-residual block once per rank`

--- PR A body: PASTE BEGIN ---

## Summary

The Kimi K3 pipeline stage keeps each attention-residual block once per pipeline rank, the memory the Attention Residuals paper states for cross-stage caching ("each block is stored exactly once across all V virtual stages", section 4.1), by passing the blocks as tensors of their own instead of stacking them.

- `kimi_k3/pipeline_parallel/stage.py`: `_assemble_stack`, `_pack_outgoing_delta` and `_split_stack_grad` work on per-block lists, so a hop sends one tensor per block and no stage copies its blocks into a fresh leaf; under the action-list runtime, a forward send is waited at the stage's own backward of that micro-batch.
- `kimi_k3/pipeline_parallel/__init__.py`: turns the forward send waits on under `_PipelineScheduleRuntime`.
- `kimi_k3/model.py`: the model takes and returns its blocks as a list.
- `kimi_k3/pipeline_parallel/PP_ATTN_RES_CACHE.md` and the stage figure's legend: one `[T, D]` tensor per block.

## Design

In the stage of #4312 a rank holds a block several times: every stage stacks the blocks it reads into a fresh `[T, N, D]` leaf that lives until its backward, the outgoing payload is another stack, and the rank cache keeps views that pin whole receive buffers and output stacks. With one tensor per block, a received block is its receive buffer and a committed block is the model's own tensor; the rank cache, the stages' inputs and the sends all reference that memory, so a block is freed once the stage that brought it onto the rank has run its backward and the forward send that carried it on has been waited. The rank cache, the routing tables and the release point are unchanged.

A pending send keeps its tensor allocated until it is waited, and torch's action-list runtime waits every send at the end of the step, which would keep every sent block until then. The stage waits a forward send at its own backward of that micro-batch, when the receiver has used it. This belongs in torch (<torch issue link>); the override goes once torch has it.

## Results

4 H100s, the Kimi K3 debug model widened to dim 6144 (17 layers in blocks of 4, 8 experts, every width that scales with dim scaled 24 times), 16 micro-batches of 2048 c4 tokens per step, Interleaved1F1B, FullAC, AdamW, seed 42, deterministic, one warm compile cache per layout. #4656 is the two commits below this PR; both trees ran 100 steps.

| layout | peak allocated per rank, #4656 (GiB) | this PR (GiB) | saved (GiB) | step time, #4656 / this PR |
|---|---|---|---|---|
| pp4 x vpp2 | 55.97 / 48.86 / 46.62 / 32.76 | 54.68 / 46.61 / 44.75 / 31.40 | 1.29 / 2.25 / 1.88 / 1.36 | 3.662 / 3.582 s |
| pp4 x vpp4 | 62.75 / 44.14 / 49.05 / 40.38 | 59.54 / 42.19 / 45.29 / 37.34 | 3.21 / 1.95 / 3.76 / 3.05 | 3.821 / 3.744 s |

Loss and grad norm equal #4656's on all 100 steps in both layouts:

| layout | step 1 | step 10 | step 50 | step 100 |
|---|---:|---:|---:|---:|
| pp4 x vpp2 | 8.08712 / 24.3750 | 7.56287 / 1040.0000 | 2.93439 / 4.9375 | 2.66764 / 1.5859 |
| pp4 x vpp4 | 8.12927 / 21.0000 | 7.97795 / 32.5000 | 2.75488 / 4.9375 | 2.43219 / 1.2266 |

The block memory the pipeline holds in step 5 (the rank cache, the stages' inputs and outputs and the pending forward sends, counted by storage) peaks at 1.78 / 2.63 / 2.32 / 1.80 GiB per rank on #4656 and 0.59 / 0.61 / 0.47 / 0.45 GiB with this PR at pp4 x vpp2, and at 3.84 / 2.81 / 4.73 / 3.49 and 0.87 / 0.77 / 0.87 / 0.77 GiB at pp4 x vpp4. The paper's bound for the same steps, each block of a micro-batch kept once until the rank's last backward of that micro-batch, is 0.52 / 0.61 / 0.52 / 0.54 and 0.73 / 0.68 / 0.82 / 0.77 GiB.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py tests/unit_tests/cpu/test_kimi_k3_attention_residual_recompute.py -q` (70 passed): the four-rank gloo pipelines of #4312, now with one tensor per block and the forward send waits, under Interleaved1F1B with the cache on and off, every block gradient bitwise with one device, and eval between steps.

## Relation to other PRs

- Stacks on #4656, which carries the blocks as a list inside the model. The first two commits here are #4656 rebased onto current main; they show here until #4656 merges.
- #4765 and #4764 stack on this PR.

--- PASTE END ---
