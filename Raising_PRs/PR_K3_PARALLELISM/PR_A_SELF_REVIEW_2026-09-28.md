# PR A 自审（2026-09-28）

用户："PR A我看了一下diff，为什么一堆乱七八糟的unit test；还有对4312合并的stage和cache的改动也太大了吧？？？我们已经raise了illustration figure PR，里面的变量名字是不能乱改的，包括4312里面的changes是被review过了，这个时候大改估计非常麻烦，并不是你写的有问题，但是这个很难推进review，你先自我审查一遍"。

审的是 `git diff aa6d9fedc 439bd2088`（PR A 自己的 diff，叠在 #4656 的列表载体提交上），5 个文件 +516/−268。插图 PR 是 #4914（`k3_pp_cache_doc` `52f9dd6d3`），它的 `PP_ATTN_RES_CACHE.md` 点名用了 4312 的这些名字和行为。

## 1. 和 #4914、和 4312 已审内容冲突的地方

| #4914 里写的 | PR A 之后 |
|---|---|
| `_assemble_stack`：收到的 delta 和 store 里的 block 拼成一个 [T, N, D] leaf | 函数删掉，改成 `_assemble` 返回 block 列表 |
| `_pack_outgoing_delta`：下一个 rank 缺的列，作为模型 stack 的视图 | 函数删掉，改成 `_outgoing_blocks` |
| `_split_stack_grad`：把 stack 的梯度拆成收到的列和存下的 block 的 deposit | 函数删掉，改成 `_route_input_grads` |
| 一跳带 hidden [T, D] 和 delta [T, Nd, D] | 一跳带 hidden 加 Nd 个 [T, D]，P2P 的元数据跟着变 |
| store 里的 block 在该 rank 最后一个 stage 的前向后释放 | 在把它带进来的 stage（bringer）的反向后释放 |
| `PPRankLocalCache`、`BlockLayoutTables`、`delta_to_send`、`commits_at`、`cache_at_entry`、`deposits_expected`、`_retrieve_recv_grads`、`attn_res_cache` | 名字都还在，行为不变 |

4312 审过的 3 个测试被删：`test_assembly_orders_blocks_and_hands_back_a_leaf`、`test_gradient_split_sends_the_received_and_deposits_the_stored`、`test_store_accumulates_deposits_and_releases_blocks_separately`。

## 2. diff 由哪几块组成

| 块 | 做什么 | 在哪 | 性质 |
|---|---|---|---|
| a | 按 block 传输：一跳每个 block 一个张量，收到的 block 就是它的接收缓冲，store 只存引用；stage 内部改成列表 | `stage.py`：删改三个 helper，`_assemble`、`_commit_and_route`、`forward_one_chunk`、`_compute_outputs`、`backward_one_chunk` 重写；新增 `_route_input_grads`、`backward_maybe_with_nosync` 覆盖 | 改 4312 的结构和名字 |
| b | 每个 block 在 bringer 的反向后释放 | `cache.release(mb, block_idxs)`、`_brought()`、`backward_one_chunk` 两行 | 小 |
| c | 接收缓冲按需分配（torch 默认给每个 micro-batch 常驻一份） | 覆盖 torch 的 `_setup_forward_recv_info`、`_setup_backward_recv_info`、`get_fwd_recv_ops`、`get_bwd_recv_ops`，用 `_make_tensor_from_meta` | 绕 torch 的行为，用私有接口 |
| d | 前向的 send 在本 stage 的反向时等掉（torch 默认拖到一步结束，work 会钉住张量） | 覆盖 `get_fwd_send_ops`，用 `_batch_p2p` | 同上 |
| e | 输入梯度的 send 在第一个能证明对方已收到的前向时等掉 | `_GradSendWaits`、`_grad_send_wait_points`（从 schedule 的 `pipeline_order` 推等待点）、覆盖 `get_bwd_send_ops`、`__init__.py` 里接线，用 `_PipelineScheduleRuntime` | 同上，而且最复杂 |
| f | 测试 | `test_kimi_k3_pp_stage.py` 删 3 加 2；`test_kimi_k3_pp_block_grads.py` 加两个记录用的子类、把 `_run_pipeline` 拆成 `_build` 加 `_run_pipeline`、新增"每个 block 由 bringer 释放"和"等待点跟着接收方走"两个测试 | 跟着 a、b、e 走 |

## 3. 省下的显存主要来自哪块（5060 实测，09-26，`PP_LOWER_BOUND_CAMPAIGN_2026-09-26.md`）

- 当时 PR A 是分两段做的：先是 V4（`9ac65f173`，叠在 4312 上的三个提交：store 直接收 delta、接收缓冲按需加前向 send 在反向时等、输入梯度 send 在证明到达时等），再是按 block 传输、列表和 bringer 释放。
- 最重的 rank：4312 11.76 → V4 7.29 → PR A 6.82 GiB。大头（−4.47 GiB）在 V4，也就是 c、d、e 三块绕 torch 的部分加上 store 收 delta；a、b 两块再省 0.47 GiB（在 stage 中间开 block 的两个 rank 上是 1.29 和 1.42 GiB）。
- V4 那一段保留了 `_assemble_stack`、`_split_stack_grad`，只把 `_pack_outgoing_delta` 改成了 `_outgoing_delta`；它和 4312 的接口更接近。
- 09-27 的 s6（#4656 对 PR A，同一个结构）是 11.47 → 7.12 GiB，没有再拆各块的贡献。

## 4. 结论

- 代码本身没有发现错误，但作为一个 PR 太难审：它同时（1）改掉 4312 审过的三个函数和三个测试、改了一跳的格式，和 #4914 的文字对不上；（2）用六个覆盖加一段 schedule 分析去绕 torch 运行时的两个行为。
- 省显存的大头在（2），而（2）按 body 自己的说法属于 torch（"they belong in torch, and the overrides go once torch offers them"）。

## 5. 可选的收法（等用户定）

1. **拆成两件事：**
   - torch 那两个行为（send 拖到一步结束才等、每个 micro-batch 常驻一份接收缓冲）先开 torch issue（草稿 `TORCH_ISSUE_PP_BUFFERS_2026-09-27.md`）；titan 这边如果等不了 torch，单开一个只含 c、d、e 的 PR，不碰 4312 的函数和测试。
   - PR A 只留 a、b 里真正需要的部分，保留 4312 的函数名（`_assemble_stack`、`_pack_outgoing_delta`、`_split_stack_grad`）、一跳的 [T, Nd, D] 格式和 4312 的测试，只改函数内部；#4914 里对应的一两句话随 PR A 一起改。
2. **PR A 回到 V4 那种形态：** 以 V4 的三个提交为底（名字基本保留），叠在 #4656 上重做，再看 a、b 还值不值得放。
3. **保持现状：** 代价是 review 要重新审 4312 的 stage，并且 #4914 合并后马上过时。

我倾向 1：最大的收益来自绕 torch 的部分，该先让 torch 知道；PR A 自己改小、名字不动，才审得动。改动量要重写后才能给准数。
