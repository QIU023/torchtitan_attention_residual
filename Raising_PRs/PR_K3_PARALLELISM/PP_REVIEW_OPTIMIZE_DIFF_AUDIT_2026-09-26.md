# pp_review_optimize 逐行审 diff（2026-09-26）

范围：`git diff 7814d1f8b 9ac65f173`，即 4312 的 head 到 pp_review_optimize 的三个提交。改了 5 个文件，+362/−78。下面的行号都是 `9ac65f173` 上的。

跑过的检查（本次）：
- `pre-commit run flake8`：通过（仓库的上限是 120 列）。
- `pre-commit run ufmt`：**不通过**，改了 3 个文件（`stage.py`、两个测试文件，+33/−12）。改动存为同目录的 `ppopt_ufmt_2026-09-26.diff`，分支没有动。
- CPU 单测：09-25 rebase 后 110 项通过，本次没有重跑。

## 0. 结论

- **行为：** 逐项推过，没有发现错误（§1）。
- **进 PR 前必须改：** µfmt、测试类的位置、注释（§2）。
- **三处靠隐含前提：** 现在是对的，但写法脆弱，要么加检查，要么换写法（§3）。
- **测试缺口：** 单 stage schedule 那条路径（payload 复制后交给 torch 发）没有任何数值测试（§5）。
- **分支没有改动。** 这些修正建议放进 PR A 的工作里一起做；PR A 会重写 store 和线上格式，§3 的三处大多会随之消失。

## 1. 逐项核对过、结论正确的地方

| 位置 | 做法 | 为什么对 |
|---|---|---|
| `stage.py:264-270`、`435-436` | action-list runtime 下，前向 send 由 stage 自己发，在本 stage 对同一 micro-batch 的反向开始时 wait | 这个反向要用下一段送回的梯度，而下一段算这个梯度之前，已经在它的前向里用过我们发的数据，所以 wait 不会卡住 |
| `stage.py:101-156`、`347-348` | 输入梯度的 send 在"证明点"wait：本 rank 上第一个前向，它的输入是接收方在消费完这个梯度之后才产出的 | NCCL 的 `wait()` 只让计算流等 NCCL 流，不阻塞 CPU。到证明点时，接收方已经收完，所以这次等待不会让流水线停住。之后计算流再复用这块显存，顺序上一定在 send 完成之后，不需要 `record_stream` |
| `cache.py:29-31` | `rows()` 经 `.data` 取 view | `.data` 的别名有自己的版本计数，`put` 写其他行时不会让已保存的 view 在反向时报版本错误。而且写入的行总在已保存的 stack 之外（stack 是前 N 行，新 block 写在第 N 行及以后） |
| `stage.py:89-98`、`333-336` | payload 改为 store 的 view，梯度经 `_PayloadFromStore` 回到模型输出 | forward 不存任何张量，backward 原样把梯度交给模型的 payload view。4 进程 gloo 测试与单卡逐位一致 |
| `stage.py:376-378`、`488-494` | store 在本 rank 第一个 stage 的反向结束时才从字典里移除 | 零气泡类 schedule 把反向拆成 input/weight 两步，weight 那一步还要用保存的 stack view。view 引用着底层存储，所以提前或推后移除字典里的引用都不影响正确性 |

## 2. 进 PR 前必须改

1. **µfmt 不通过。** 3 个文件要重新格式化，改动见 `ppopt_ufmt_2026-09-26.diff`。其中 `stage.py:122` 的生成器写成了一行（106 列），`_PayloadFromStore.forward` 的签名（`stage.py:93`）也超长。
2. **新测试类放在 `if __name__ == "__main__":` 之后**（`test_kimi_k3_pp_stage.py:166-199`）。pytest 能收集到它，直接运行这个文件时 `unittest.main()` 在类定义之前就执行了，这个测试不会跑。要挪到 main guard 之前。
3. **注释按"默认不加注释"规则过一遍。** 新增 22 行注释和 docstring，逐条判断：

| 位置 | 内容 | 处理 |
|---|---|---|
| `stage.py:9-10` | 模块 docstring 加的线上格式说明 | 删第 10 行（格式由 `_outgoing_delta` 的代码表达）；后缀说明沿用文件原有的约定，保留 |
| `stage.py:226-227` | "Receive buffers are allocated when … torch's default keeps one per micro-batch" | 删：讲设计理由，放 PR body |
| `stage.py:267-268` | 为什么在反向时 wait 是安全的（两行） | 压成一行，只写约束："the receiver has used the tensors once this stage's backward of the micro-batch starts" |
| `stage.py:273` | 为什么单 stage schedule 下 payload 要复制 | 保留一行：否则 send 会把整块 store 钉到 step 末 |
| `stage.py:279` | 证明点 | 保留一行 |
| `stage.py:306` | "The delta was received into its rows; the stack is a view …" | 删：复述下一行代码 |
| `stage.py:334` | "The blocks sit in the store, so …" | 删：设计理由 |
| `stage.py:376` | "Stacks are views of the store, so …" | 删，并按 §4 第 3 条简化条件 |
| `stage.py:408` | 为什么梯度可以原地加 | 保留一行：它依赖接收缓冲按需分配，改回常驻缓冲就会出错 |
| `stage.py:209` | 只有 cache 开时才就地接收 | 保留一行 |
| `cache.py:30` | `.data` 的原因 | 保留 |
| `cache.py:34`、`42` | `stack`、`mark` 的 docstring | 删：#4577 的标准是私有辅助不写 docstring |
| `__init__.py:121` | 为什么只在 action-list runtime 下自持 send | 保留一行：单 stage schedule 会把 send 和 recv 合成一批以避免死锁，自持 send 会破坏这一点 |

## 3. 靠隐含前提的写法

1. **接收方默认 delta 是连续的几行**（`stage.py:255-257`）：`first = delta_blocks[0]`，然后接收 `rows(first, K)`，没有检查 `delta_blocks == range(first, first + K)`。
   - 现在是对的：发送方的 `_outgoing_delta`（`stage.py:62-66`）检查了 delta 是 stack 的尾部，两端用的是同一张 layout 表。
   - 但接收方自己不检查。如果以后 layout 改了，数据会写进错误的行，而 `_assemble` 的检查（`stage.py:309`）仍然会通过，因为它只看行号集合是不是从 0 开始连续。
   - 改法：在接收方加同样的检查。按 block 分配之后，这个前提本身就没有了。
2. **store 的 dtype 取自 hidden 的接收缓冲**（`stage.py:254`），不是 delta 自己的 `tensor_meta`。
   - K3 里两者 dtype 相同，所以现在没问题。一旦不同（比如 hidden 以 fp32 输出、block 是 bf16），NCCL 会按 delta 的字节数写进 dtype 不同的缓冲。
   - 改法：用 `delta_info.tensor_meta` 分配，或者两者不一致时报错。
3. **靠 `op.tensor.dim() == 3` 认出 payload**（`stage.py:272`）。这依赖 hidden 是二维的 [T, D]。改法：按输出的位置认（第 1 个输出是 payload）。

另有一处小问题：`_placeholder`（`stage.py:222-224`）在拿不到参照张量时默认用 `torch.bfloat16`，可以改成沿用 `tensor_meta` 的 dtype。

## 4. 效率与结构

1. **`_setup_forward_recv_info`、`_setup_backward_recv_info` 仍会先让 torch 分配 M 份缓冲再丢掉**（`stage.py:229-244`）。
   - 显存回到缓存池，后面的分配可以复用，所以不是泄漏。
   - 但建立阶段会有一个 M 份缓冲的瞬时峰值，缓存池的 reserved 也被撑大一次，大段显存可能留下碎片。
   - 彻底避免要在 torch 里改（见第 4 条）。
2. **store 在 RECV_F 发出时就分配**（`stage.py:254`）。runtime 会把接收提前发出，所以 store 比前向更早占上显存。量不大，按 block 分配时顺便改到 bringer 的前向。
3. **`stage.py:376-378` 的释放条件对显存没有影响。** 已保存的 view 引用着底层存储，字典里的引用早去晚去都一样。可以删掉这个条件，恢复 4312 原来"最后一个前向后移除"的写法，少一个分支。
4. **torch 私有接口：**
   - 新重写的私有方法：`_setup_forward_recv_info`、`_setup_backward_recv_info`；
   - 新重写的方法：`get_fwd_recv_ops`、`get_fwd_send_ops`、`get_bwd_send_ops`、`get_bwd_recv_ops`；
   - 新用到的私有名字和字段：`_batch_p2p`、`_make_tensor_from_meta`、`_ComputationType`、`_PipelineScheduleRuntime`，以及 `args_recv_info`、`grad_recv_info` 的 `buffer` 和 `tensor_meta` 字段。
   其中"接收缓冲按需分配"和"send 提前 wait"对任何 PP 模型都成立，按计划文档（`PP_LOWER_BOUND_PLAN_2026-09-25.md` §2、§6）先向 torch 提 issue。

## 5. 测试

1. **单 stage schedule 那条路径没有数值测试。** 4 进程 gloo 的 block 梯度测试只跑 `ScheduleInterleaved1F1B`，也就是 action-list runtime 那条路径。1F1B、GPipe 下 payload 复制后交给 torch 发的那条路径（`stage.py:271-275`），只有 `TestStageSwap.test_single_stage_schedule` 碰到，而它只测 stage 替换，不比数值。应该在 block 梯度测试里加一个单 stage schedule 的变体。
2. **没有测试直接断言显存行为：**
   - delta 落进 store 的行（比较 data_ptr）；
   - 读完之后接收缓冲换成了占位张量。
   建议在 `test_kimi_k3_pp_stage.py` 里用单进程 gloo 加两个假 stage，各加一条断言。
3. **`TestGradSendWaitPoints` 的写法：**
   - 用 `ScheduleInterleaved1F1B.__new__` 建 schedule，再手动设 5 个私有字段，然后调 `_calculate_single_rank_operations`，和 torch 内部实现绑得很紧；
   - 在函数里 import 了 `_ComputationType` 和 `_grad_send_wait_points`，而模块顶部已经从 stage 里 import 了东西。
   可以改用真实构造：块梯度测试里已有真实的 schedule，直接从 `schedule.pipeline_order` 取。
4. **`test_kimi_k3_pp_block_grads.py:21-27` 对同一个模块有两条 import，要合并；`:203` 的 `isinstance` 放在了循环里面。**

## 6. 与计划的关系

- 这里的 §3 第 1、2 条和 §4 第 2、3 条，PR A（按 block 传输和存储，在 bringer 反向时释放）会一并改掉。
- §2 和 §5 与 PR A 无关，做 PR A 时先修。
- §4 第 1、4 条属于 torch 的 issue。
