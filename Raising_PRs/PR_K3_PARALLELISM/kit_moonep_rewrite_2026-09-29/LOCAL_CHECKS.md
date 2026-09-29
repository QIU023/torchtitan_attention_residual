# MoonEP 重写的本地检查（8 × RTX 5060 Ti，2026-09-29）

这台机器没有 NVSwitch，跑不了真的 MoonEP。所以另写了一个仿 MoonEP 公开版（`33327eb`）API 的假包（`local/fake_moonep/`），只用来验证 titan 这一侧的集成。

- **接口和检查照抄公开版：** `Buffer` 的 `dispatch`/`combine`/`prefetch_weight`/`reduce_grad` 签名，`_validate_rank_strided_pool` 以及预取、规约里的形状和连续性断言；
- **语义用 EP 组上的 NCCL 集合通信实现：**
  - 每个 rank 的槽固定放下一个 rank 的专家；
  - 每份 token 按奇偶发到专家的本家或持有副本的槽；
  - 每组按 128 行补零；
- **池的初值是 NaN：** titan 只要读到没写过的行，结果里就会冒出 NaN。

这里的结果只说明 titan 的集成逻辑对。它们不是 MoonEP 的数字，不进 body。

## 找到并修掉的两个 bug

1. autograd 的 `ctx.metadata` 是保留属性，赋值直接报错。改成 `ctx.dispatch`。
2. 反向里从 SPMD 上下文取 EP mesh，拿到的是 None：反向可能在另一个线程上跑，那里没有这个上下文。改成前向时把 group 和 buffer 存进 ctx，DeepEP 也是这么做的。

两个都是 GPU 上第一次跑就会暴露的错，gloo 或 CPU 测试看不到。

## 测试

- CPU（`test_moe.py`、`test_inference_moe.py`、`test_integration_test_definitions.py`、`test_expert_parallel.py`）：53 passed，13 subtests passed（`local/cpu_tests.log`）。
- `tests/unit_tests/gpu/test_moonep.py`，接假包、两卡、去掉 skipif 以外原样：2 passed（`local/unit/test_moonep_fake.log`）。
  - 热点路由（所有 token 都发到 rank 0 的专家）和均匀路由两种情况下，输出、输入梯度、`w13`/`w2` 的梯度都和 fp32 稠密参考一致；
  - 热点路由下，token 确实到了槽里。
- scoped pre-commit：除 pyrefly 外全部通过；pyrefly 报的 47 个错和同一个 main（`5dc97a3e7`）上的基线逐条相同（`local/precommit.log`）。hook 顺手改动的 25 个无关文件已还原。

## 端到端（h100 格子 `kimi_k3_moonep_fsdp4_ep4`，4 卡，接假包）

对照：同一个格子换成 standard 后端。`--debug.seed 42 --debug.deterministic`，同一份预热缓存，各 10 步。脚本在 `local/e2e/`；本机的旧 torch 需要打 `shim_5dc97a3e7.patch`（只给 `pipeline_per_edge_p2p` 加 `hasattr`），跑前打上、跑完撤掉。

| step | standard loss / grad norm | MoonEP（假包）loss / grad norm | loss 相对差 |
|---:|---:|---:|---:|
| 1 | 7.99519 / 2.4531 | 7.99519 / 2.4531 | 0.0e+00 |
| 2 | 7.76125 / 2.5469 | 7.76095 / 2.5469 | 3.9e-05 |
| 3 | 7.63983 / 3.9062 | 7.63723 / 3.9219 | 3.4e-04 |
| 4 | 7.30177 / 4.6562 | 7.29913 / 4.7188 | 3.6e-04 |
| 5 | 6.80197 / 7.0000 | 6.80544 / 6.9688 | 5.1e-04 |
| 6 | 6.76491 / 6.7812 | 6.75990 / 6.7812 | 7.4e-04 |
| 7 | 6.05292 / 7.2188 | 6.05035 / 7.2500 | 4.2e-04 |
| 8 | 6.17476 / 7.8438 | 6.17868 / 7.8438 | 6.3e-04 |
| 9 | 5.94753 / 8.1250 | 5.93580 / 8.1875 | 2.0e-03 |
| 10 | 5.67797 / 8.0000 | 5.68106 / 8.0625 | 5.4e-04 |

峰值显存（GiB，各 rank）：standard 0.25, 0.24, 0.24, 0.24；MoonEP（假包）0.28, 0.27, 0.27, 0.27。

噪声底：同一个 standard 格子、同一个种子，第一轮和第二轮各自预热了一份编译缓存，第 1 到 10 步的 loss 相对差是 0.0e+00 6.1e-05 3.2e-04 3.5e-04 3.0e-04 2.3e-04 1.4e-04 4.6e-04 2.4e-03 1.9e-03。所以 MoonEP（假包）对 standard 的差，和 standard 自己换一份缓存的差是同一量级。第 1 步两者都逐位相同。

第一轮（修 dispatch 直接返回 plan 之前的代码，另一份缓存）的结论相同：第 1 步逐位相同，之后最大相对差 1.2e-3。
