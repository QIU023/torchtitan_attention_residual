# PR A body v4（最小 diff 版，保留 4312 的结构），2026-09-30

## 状态（不粘贴）

- **10-01 Design 改成两级 bullet（用户："这一段话写成简洁 bullet points之类的格式，然后打印并且覆盖到PR A body"）：** 内容和原来两段一致，没有增减事实；GitHub 上 #4963 的 body 需要用粘贴区整段替换。
- **10-01 数值表列到第 100 步（用户："我的理解是我们要列100步，还要50步"）：** 成对表的列改成第 1 / 10 / 50 / 100 步加"100 步中逐位相同的步数"，和 09-21 的数值表规则一致（C4 数据列到第 100 步）。数从 H100 原始日志重新解析（`kit_pra_min_2026-09-30/tab_numerics.py`，按 rank 分别解析，打印真实 loss 的 rank 每一步都一致），五个布局都是 100 / 100，第 50、100 步的数和最早那张表一致。
  - 用户 10-01：第 1 步逐参数梯度和噪声底这两项不需要，粘贴区里的噪声底那一行、第 1 步梯度那一句和表头那句说明已删。H100 上排的 `kit_h100_2026-09-30/pra/run_pra_floor_grads.sh` 可以撤掉，没撤也不用它的结果。
  - 回复 reviewer 时可用的要点（不进粘贴区）：本 PR 只改块存在哪里（拷贝变成引用），每个运算的输入值不变；梯度累加的顺序理论上可能变，所以不靠推理，靠实测：第 1 步 loss 和逐参数梯度逐位相同，五个布局 100 步全部逐位相同，噪声底那一行说明这个结果不是平凡相等。
  - reviewer 可能会问：pp4 × vpp2 的参照树自己在第 10 步 grad norm 就是 1040（两棵树相同，是这套设置本身的尖峰）；dp2 × pp2 × vpp2 第 100 步 grad norm 15.75 也是两棵树相同。被问到再答，不写进正文。

- **10-01 数值表改成成对的（用户："Loss and grad norm equal #4656's ... 没有和原始PR的对比表吗？如何说服reviewer它数值没问题？？"）：** 原来只列了 PR A 一行，加一句"和 #4656 相等"，而且列了第 50、100 步。现在按数值表规则改：每个布局"不含本 PR / 本 PR"成对两行，同一份暖 cache、同 seed 同数据（H100 原始日志 `kit_h100_2026-09-30/results/pra_h100*/` 重新解析，dp2 × pp2 × vpp2 按 rank 分开解析，五个布局都是 100/100 逐位相同）；只列第 1 / 10 / 20 步，"100 步全同"用计数列表达，后面的步不列。
  - 还缺两项，都要 GPU（5060 也行，只是要和表的设置一致，最好在 H100 上）：
    1. 噪声底一行：#4656 在 pp4 × vpp2 用一份新的 compile cache 再跑 20 步，看换 cache 本身让第 3 / 10 步变多少。没有这一行，reviewer 不知道"逐位相同"是不是平凡的。
    2. 第 1 步逐参数梯度：pp4 × vpp2 两棵树在同一份 cache 上 dump 第 1 步所有参数的梯度，数 N / N 逐位相同。这是 CLAUDE.md 里的正确性标准（第 1 步 loss 逐位 + 第 1 步梯度逐位、按参数计数）；现在只有 grad norm 这个标量。
- **10-01 复测成立，粘贴区已改（用户："PR A的cache内存下界结果可能不准确，重新按照这个修复"）：**
  - 探针：`kit_h100_2026-09-30/pra/probe_bound.py` 的 `tracking` 把前向 send 的第 0 个 op 记成 hidden（断言它就是输出 0），不再算进 `block_gib`，另报 `hidden_sends_alive_gib`，以及 `hidden_sends_only_gib`（存储不在任何 stage 输出里的 hidden send）。
  - 5060 复测（实测，`kit_pra_min_2026-09-30/run_bound_5060.sh`，结果在 `kit_pra_min_2026-09-30/results_5060_1001/`）：T1d 的设置（dim 2048，16 × 2048 个 c4 token，FullAC，seed 42，deterministic，20 步，第 5 步逐动作记账），pp4 × vpp2，`e66a9442b` 对 `312bc8144`，两棵树各用同一份暖 cache 的拷贝。
    - PR A 的块数：rank 0 到 2 的 64 个动作全部等于紧界；rank 3 有 48 个相等，另外 16 个是最后一个 stage 的前向（7F0 到 7F15），比紧界少 1 块。原因是紧界把最后一个 stage 在前向里完成、当场用掉的那个块也算了，实际没人存它。
    - PR A 只被 send 扣住的块：每个动作都是 0。
    - 待等的 hidden send：PR A 最多 10 / 8 / 6 / 4 个，#4656 32 / 32 / 32 / 16 个。两棵树每个动作上都等于按调度数出来的个数：非最后 stage 的每个前向发一个；#4656 到步末才等，PR A 在发送方自己的反向时等。PR A 的 hidden send 一直还被 stage 输出持有（`hidden_sends_only` 为 0）；#4656 在反向之后只剩 send 扣着（32 / 32 / 32 / 16）。
    - 20 步 loss 和 grad norm 两树相同。
  - H100 的按块数（推算，`kit_pra_min_2026-09-30/h100_blocks_from_old.py`，输出 `h100_blocks_derived.md`）：旧记录的 `block_gib` 减去按调度数出的待等 hidden 个数乘一个 [T, D]。PR A 上它和"旧 block 减 sends only"每个动作都一致；pp4 × vpp2 两棵树的块数和 5060 实测完全相同（块数只由调度决定，和宽度无关）。PR A 在五个布局的每个动作上都等于紧界，只有最后 stage 的前向少 1 块（pp4 × vpp4 没有这种动作）。
  - 粘贴区改了两处：
    - Results 最后一段换成按块计的表：#4656、本 PR、紧界、论文界，两棵树待等的 hidden send 各一列。整卡峰值和每卡省的 GiB 不变。
    - Design 那句改回 09-30 之前的原句（块在带进来的 stage 反向后释放）。09-30 加的"并且把它继续发出去的前向 send 也等完"是照同一个探针假象加的。按块实测，块就在紧界上释放：继续往下发它的 send 由同一 rank 上序号不小于它的 stage 发出，在那个 stage 的反向时等完，不会晚于带进来的 stage 的反向。
  - H100 上要不要用修正后的探针直接重测这几个数，你定。body 里写的是 H100 的数，没有引 5060。
- **10-01 已核实（见上一条）：H100 上"超出紧界"的部分是 hidden state，不是块。**
  - 原因：`pra_bound/probe_bound.py` 的 `tracking` 把每个前向 send 的所有张量都记进 `_SENT`，包括第 0 个 op（hidden state）。`_account` 统计输出时去掉了 `out_tuple[0]`，所以还没等的 hidden send 没有被别处持有，会落进 "sends only"，再被算进 `block_gib`。
  - 本机旁证（`kit_pra_min_2026-09-30/cpu_bound.py`，新加环境变量 `SKIP_HIDDEN`，`345e9e00a` 的树，pp4 × vpp2）：`SKIP_HIDDEN=1` 时四个 rank 超出紧界都是 0；`SKIP_HIDDEN=0` 时是 10 / 8 / 6 / 4 个 [T, D]。H100 pp4 × vpp2 超出紧界 0.24 / 0.18 / 0.14 / 0.10 GiB，按 dim 6144、seq 2048、bf16 一个 [T, D] 是 0.0234 GiB 折算，是 10.3 / 7.7 / 6.0 / 4.3 个，和本机一致。
  - 5060 复测（不需要 H100）：`probe_bound.py` 的 `tracking` 里改成 `for op in list(ops)[1:]`（或者把第 0 个 op 单独记成 hidden），用 T1d 的设置跑 pp4 × vpp2，#4656 `e66a9442b` 对 PR A `312bc8144`，第 5 步逐动作记账。预期 PR A 的块占用在每个动作上都等于紧界。如果成立，Results 里"落在论文界上"那段改成按块算的数，hidden send 另列。整卡峰值的数字不受影响。
- **10-01 压成一个提交并推送（用户："压到一个之后push到PR分支和review分支"）：** #4963 的 head 分支就是 `pp_review_optimize`（PR 分支和 review 分支是同一个）。`36cfddf87` + `345e9e00a` 压成 `312bc8144`（父提交是 #4656 的 `e66a9442b`，树和 `345e9e00a` 逐字相同，提交信息用 `8b0fcbe38` 那条，没有 trailer），force-with-lease 推送；旧 head 备份在 `backup/pp_review_optimize_pre_20260930b`（`..._20260930` 已经是 `5a8163a58`）。#4963 现在是 #4656 的两个提交加这一个。GitHub 上的 body 还是旧的，要用下面的粘贴区替换。#4765、#4764 还叠在 `36cfddf87` 上，下一步重叠。
- **09-30 H100 结果已填进粘贴区（`PRA_H100_2026-09-30.md`）：** Results 放了 dim 6144 的两个 4 卡布局（100 步逐位一致，每卡省 1.3 到 3.8 GiB，步时快约 2%，块占用落在论文界上）。dp2 × pp2 × vpp2、pp2 × vpp2、pp2 × vpp4 在 6144 下两个树都 OOM，dim 5120 补跑的三行也已填上（100 步逐位一致，每卡省 1.0 到 2.1 GiB）。Design 改了一句：原文"块在带进来的 stage 反向后就释放"，实测块占用在论文界上、没到紧界，差的是前向 send 要到发送方自己反向时才等，所以改成"带进来的 stage 反向完、并且把它继续发出去的前向 send 也等完之后释放"。要不要这样写你定。
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

- Before this PR (the stage of #4312), a rank holds each block several times:
  - every stage stacks the blocks it reads into a fresh `[T, N, D]` leaf that lives until its backward;
  - the outgoing payload is a second stack;
  - the rank cache keeps views that pin whole receive buffers and output stacks.
- With this PR, one tensor per block:
  - a received block is its receive buffer, and a committed block is the model's own tensor;
  - the rank cache, the stages' inputs and the sends all reference that memory, so a block is freed once the stage that brought it onto the rank has run its backward;
  - the rank cache, the routing tables and the release point are unchanged.
- Send waits:
  - a pending send keeps its tensor allocated until it is waited, and torch's action-list runtime waits every send at the end of the step;
  - the stage waits a forward send at its own backward of that micro-batch, when the receiver has used it;
  - this belongs in torch (<torch issue link>), and the override goes once torch has it.

## Results

4 H100s, the Kimi K3 debug model widened to dim 6144 (17 layers in blocks of 4, 8 experts, every width that scales with dim scaled 24 times), 16 micro-batches of 2048 c4 tokens per step, Interleaved1F1B, FullAC, AdamW, seed 42, deterministic, one warm compile cache per layout. #4656 is the two commits below this PR; both trees ran 100 steps. The layouts with two pipeline ranks hold half the model per GPU and run the model at dim 5120, since at 6144 both trees run out of memory.

| layout | peak allocated per rank, #4656 (GiB) | this PR (GiB) | saved (GiB) | step time, #4656 / this PR |
|---|---|---|---|---|
| pp4 x vpp2 | 55.97 / 48.86 / 46.62 / 32.76 | 54.68 / 46.61 / 44.75 / 31.40 | 1.29 / 2.25 / 1.88 / 1.36 | 3.662 / 3.582 s |
| pp4 x vpp4 | 62.75 / 44.14 / 49.05 / 40.38 | 59.54 / 42.19 / 45.29 / 37.34 | 3.21 / 1.95 / 3.76 / 3.05 | 3.821 / 3.744 s |
| dp2 x pp2 x vpp2, dim 5120 | 52.51 / 52.51 / 39.28 / 39.28 | 51.26 / 51.26 / 38.27 / 38.27 | 1.25 / 1.25 / 1.02 / 1.02 | 4.834 / 4.816 s |
| pp2 x vpp2, dim 5120 | 67.85 / 49.77 | 66.60 / 48.75 | 1.25 / 1.02 | 5.231 / 5.271 s |
| pp2 x vpp4, dim 5120 | 67.31 / 52.72 | 65.69 / 50.65 | 1.62 / 2.07 | 5.345 / 5.252 s |

Loss / grad norm with and without this PR (without it is the tree of the two #4656 commits below, whose pipeline stage is main's), each pair on one shared warm compile cache with the same seed and data; the last column counts the steps of the 100 whose loss and grad norm are identical.

| layout | tree | step 1 | step 10 | step 50 | step 100 | steps identical to without |
|---|---|---:|---:|---:|---:|---:|
| pp4 x vpp2 | without this PR | 8.08712 / 24.3750 | 7.56287 / 1040.0000 | 2.93439 / 4.9375 | 2.66764 / 1.5859 | |
| pp4 x vpp2 | this PR | 8.08712 / 24.3750 | 7.56287 / 1040.0000 | 2.93439 / 4.9375 | 2.66764 / 1.5859 | 100 / 100 |
| pp4 x vpp4 | without this PR | 8.12927 / 21.0000 | 7.97795 / 32.5000 | 2.75488 / 4.9375 | 2.43219 / 1.2266 | |
| pp4 x vpp4 | this PR | 8.12927 / 21.0000 | 7.97795 / 32.5000 | 2.75488 / 4.9375 | 2.43219 / 1.2266 | 100 / 100 |
| dp2 x pp2 x vpp2, dim 5120 | without this PR | 8.06354 / 24.3750 | 6.34372 / 14.5000 | 2.74512 / 3.9688 | 2.25450 / 15.7500 | |
| dp2 x pp2 x vpp2, dim 5120 | this PR | 8.06354 / 24.3750 | 6.34372 / 14.5000 | 2.74512 / 3.9688 | 2.25450 / 15.7500 | 100 / 100 |
| pp2 x vpp2, dim 5120 | without this PR | 8.06628 / 23.1250 | 5.24328 / 22.1250 | 2.80749 / 4.2188 | 2.45762 / 1.7578 | |
| pp2 x vpp2, dim 5120 | this PR | 8.06628 / 23.1250 | 5.24328 / 22.1250 | 2.80749 / 4.2188 | 2.45762 / 1.7578 | 100 / 100 |
| pp2 x vpp4, dim 5120 | without this PR | 8.11475 / 35.2500 | 6.92955 / 23.1250 | 2.77905 / 4.0625 | 2.40688 / 1.1328 | |
| pp2 x vpp4, dim 5120 | this PR | 8.11475 / 35.2500 | 6.92955 / 23.1250 | 2.77905 / 4.0625 | 2.40688 / 1.1328 | 100 / 100 |

Blocks held in step 5 at each rank's peak, in `[T, D]` blocks (24 MiB at dim 6144, 20 MiB at dim 5120): the rank cache, the stages' saved inputs and outputs and the pending forward sends of blocks, counted by storage. The tight bound frees each block at the backward of the stage that brought it onto the rank; the paper's bound keeps every block of a micro-batch until the rank's last backward of that micro-batch. With this PR the blocks equal the tight bound at every forward and backward of the step, except the last stage's forwards at pp4 x vpp2 and in the pp2 layouts, which sit one block below it because the bound also counts the block that stage completes and uses inside its forward.

| layout | #4656 | this PR | tight bound | paper bound | pending hidden-state sends, #4656 | this PR |
|---|---|---|---|---|---|---|
| pp4 x vpp2 | 44 / 81 / 69 / 62 | 15 / 18 / 14 / 15 | 15 / 18 / 14 / 15 | 22 / 26 / 22 / 23 | 32 / 32 / 32 / 16 | 10 / 8 / 6 / 4 |
| pp4 x vpp4 | 100 / 58 / 143 / 102 | 19 / 17 / 23 / 21 | 19 / 17 / 23 / 21 | 31 / 29 / 35 / 33 | 64 / 64 / 64 / 48 | 18 / 16 / 14 / 12 |
| pp2 x vpp2, dim 5120, with or without dp2 | 38 / 43 | 9 / 9 | 9 / 9 | 13 / 13 | 32 / 16 | 4 / 2 |
| pp2 x vpp4, dim 5120 | 34 / 81 | 9 / 11 | 9 / 11 | 15 / 17 | 64 / 48 | 8 / 6 |

A hidden-state send carries a stage's output, one `[T, D]` that is not a block, so it is counted apart (one per forward of a stage other than the last, taken out of the step 5 records); its memory is real and is part of the peaks above. With this PR it is waited at the sender's own backward of that micro-batch; on #4656 it stays until the end of the step.

## Test plan

- `pytest tests/unit_tests/cpu/test_kimi_k3_pp_block_grads.py tests/unit_tests/cpu/test_kimi_k3_pp_stage.py tests/unit_tests/cpu/test_kimi_k3_pp_layout.py tests/unit_tests/cpu/test_pipeline_parallel.py tests/unit_tests/cpu/test_kimi_k3_attention_residual_recompute.py -q` (70 passed): the four-rank gloo pipelines of #4312, now with one tensor per block and the forward send waits, under Interleaved1F1B with the cache on and off, every block gradient bitwise with one device, and eval between steps.

## Relation to other PRs

- Stacks on #4656, which carries the blocks as a list inside the model. The first two commits here are #4656 rebased onto current main; they show here until #4656 merges.
- #4765 and #4764 stack on this PR.

--- PASTE END ---
