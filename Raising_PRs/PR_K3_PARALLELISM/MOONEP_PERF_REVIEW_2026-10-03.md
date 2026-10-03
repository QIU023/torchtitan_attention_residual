# MoonEP #4751：第二轮云端审查（性能九条）的核实和盲写（2026-10-03）

用户原话："看起来具体的问题特别多 逐个验证然后在moonep review分支全盲写 明天给你h100测试"。

依据：MoonEP `33327eb` 的源码和 README（`scratchpad/MoonEP`），K3 技术报告 §5.2.1 和 §5.2.2（用户上传的 PDF），titan 的 `torch._grouped_mm` 求导公式（`FunctionsManual.h`），以及实测。

## 逐条结论

| # | 意见 | 核实 | 处理 |
|---|---|---|---|
| 1 | 反向强制重算专家前向（12 次 GEMM 对 9 次） | 成立。报告 §5.2.2 也是不重算 GEMM，只对 element-wise 算子做重算。 | c1：op 额外输出 gate/up（不可求导）并保存；反向只重填、重算激活，按 autograd 的 `_grouped_mm` 求导公式手写 6 个 GEMM。和旧版**逐位一致**。代价是多存 gate/up；它们是 MUST_SAVE op 的输出，开 AC 时也会一直持有。 |
| 2 | 梯度经 fp32 池整拷 | 成立。`reduce_grad` 的 `local_*_grad` 只要求连续 fp32，不要求放在 VMM 里；kernel 只是累加进去。 | c1：只有 slot 行进 fp32 池（池从 2·epn 行减到 epn 行）；本地行 cast 一次后直接交给 `reduce_grad`。逐位一致。 |
| 3 | 反向重填同步 | 重填躲不开：MoonEP README 说预取池本来就是所有层共享的。审查要的重叠对象是共享专家的反向。 | c3：共享专家在 side stream 上，它的反向由 autograd 放在同一 stream，和主 stream 上的重填重叠。 |
| 4 | 前向 prefetch 零重叠 | 成立。报告 §5.2.1："For the shared experts, we dispatch their GEMMs to a separate stream"。 | c3：`MoE.Config.shared_experts_stream`（默认关）。共享专家先在 side stream 上发出，和 router、dispatch、prefetch、路由 GEMM 重叠，求和前汇合。 |
| 5 | `reduce_grad` 同步 | 成立。有了手写反向，可以在 op 内部安全地做：只碰 MoonEP 的 VMM 和 MoonEP 自己 record_stream 过的 local 梯度，返回前等 event，下一层写 slot 前仍有 dispatch 反向的入口栅栏。 | c4：先算 wgrad，`reduce_grad(async_finish=True)`，同时算 dx 的两个 GEMM，再等 event。和同步版逐位一致。 |
| 6 | dispatch/combine 边界拷贝 | 方向对，但现在做不了：dispatch 是 ORDERED 的 op，SAC 会缓存它的输出，零拷贝视图会被下一次 dispatch 覆盖。报告的做法是反向再 dispatch 一次拿 x，加上零拷贝。 | 暂不做。要把三个 op 合成一个才行。 |
| 7 | 路由权重的 fp32 绕路 | 前提不对：titan 的标准 dispatcher 也是 `(out.to(float32) * scores).to(dtype)`，fp32 中间张量标准路径一样有。真正可省的是为 prob 梯度保存的 fp32 输出，报告 §5.2.2 改写了这个梯度。 | c2：权重在 expert op 里乘（前向仍是 fp32 乘、转 bf16，逐位不变），反向用 `<grad_out @ W_down, hidden>` 求权重梯度，不保存 output。梯度只在 bf16 舍入级变化，对 fp32 参考的误差和旧版同一量级。 |
| 8 | 本地行永久副本 | 成立。MoonEP README 写的是本地行直接 alias 参数，但那需要 bf16 参数放在 VMM 内存里，FSDP 自己分配参数的展开存储。 | 记下，属于配方级决定。 |
| 9 | fp32 梯度池只需 slot 行 | 成立。 | 随 c1 一起做了。 |

## review 分支（未推 PR 分支）

`moonep_review1` = `1db6fbcd1`，在 PR head `5e4596dc7` 上：

- `a7864a5a3` c1
- `06d8c338d` c2
- `493ce26e0` c3
- `1db6fbcd1` c4

这几个提交是重建过的：在 `722836cc6` 系列的基础上只改了命名、docstring，去掉了 DTensor 辅助函数，hook 改成具名函数，共享专家的返回类型改成 pyrefly 能收窄的写法，不改计算。pyrefly 和 PR head 一样，是同一组 17 个错误，没有新增。PR 分支 `k3_moonep_seam` 不动。

## 本地验证（8 × 5060，假 MoonEP；结果不进 body）

- **假包扩展**：`scratchpad/fake_moonep_async`，`reduce_grad` 支持 `async_finish`，在自己的 comm stream 上跑，可以用 `FAKE_MOONEP_ASYNC_SLEEP` 延迟。
- **逐位比对**（`kit_moonep_perf_2026-10-03/bitwise_dump.py`，6 个场景：单层；两层两 micro-batch 分别配 none / selective / full / region AC；4 卡双热点 home）：
  - old 对 c1：全部逐位相同。
  - c3 对 c4：全部逐位相同，加 2 亿周期延迟后也相同。
  - 去掉 `wait_event` 的变异版：梯度完全错。
  - old 对 c2：output 逐位相同。单层时 `grad_w2` 逐位相同，`grad_x`、`grad_w13`、权重梯度差 2e-3 到 1e-2（相对最大值）。对 fp32 参考的最大误差两版互有高低，例如两层两 micro-batch：`grad_x` 1.56e-3 对 1.83e-3，`grad_w13` 1.09e-2 对 8.51e-3。
- **单测**：
  - `test_moonep.py` 9 passed（假包，含新加的路由权重梯度检查）；同一份新单测在 old 上也 9 passed。
  - `test_moe_shared_experts_stream.py`：输出和参数梯度逐位相同，`x` 的梯度在容差内（共享专家先发出，三路梯度累加顺序变了）。去掉前向汇合的变异版会失败（测试里人为拖慢了共享专家）。
  - CPU：MoE、transforms、各模型 MoE 测试共 108 passed。

## 端到端（4 × 5060，假包，debug 模型，10 步，同一份预热缓存）

Kimi K3 debug 模型，FSDP 4 × EP 4，seq 512，deterministic。每棵树先用一份缓存预热，每个格子在这份缓存的副本上跑。old 树是 `5e4596dc7`；new 树是 `722836cc6`，和最终的 `1db6fbcd1` 逐位相同，6 个场景都比过。表在 `kit_moonep_perf_2026-10-03/results_5060/`。

| 格子 | 第 1 步 loss / grad norm | 第 10 步 | 对 old 标准 EP 的最大相对差 | 峰值 GiB |
|---|---:|---:|---:|---:|
| old 标准 EP | 7.99561 / 2.4531 | 5.68613 / 8.0625 | | 0.22 |
| old MoonEP | 7.99561 / 2.4531 | 5.67827 / 7.9688 | 1.38e-3 | 0.28 |
| old MoonEP FullAC | 同 old MoonEP | 同上 | 1.38e-3 | 0.24 |
| new 标准 EP | 和 old 标准 EP 10 步完全相同 | | 0 | 0.22 |
| new 标准 EP 加共享专家 stream | 和标准 EP 10 步完全相同 | | 0 | 0.22 |
| new MoonEP | 7.99561 / 2.4531 | 5.67305 / 8.0000 | 2.30e-3（对 old MoonEP 是 1.81e-3） | 0.32 |
| new MoonEP FullAC | 同 new MoonEP | 同上 | 2.30e-3 | 0.28 |
| new MoonEP 加共享专家 stream | 和 new MoonEP 10 步完全相同 | | 2.30e-3 | 0.32 |

- 第 1 步所有格子相同：前向没变。
- new MoonEP 对 old MoonEP 的差（1.81e-3）来自 c2 的舍入顺序，和 MoonEP 对标准 EP 本来的差在同一量级。这是 5060 加假包上的数，不进 body；H100 上要重测。
- 峰值多出的 0.04 GiB 是 c1 存下的 gate/up。在 FullAC 下这部分也一样持有，因为它们是 MUST_SAVE op 的输出。
- microbench 在两棵树、experts 和 moe 两种模式下都冒烟通过（假包上的耗时没有意义）。

## H100 计划

见 `kit_moonep_perf_2026-10-03/README.md`。
