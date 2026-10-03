# MoonEP 性能轮（2026-10-03 写，计划 10-04 上 H100）

review 分支 `moonep_review1` 在 PR head `5e4596dc7` 上加了四个提交，都是盲写，本地只在 8 × 5060 上用假 MoonEP 包验证过：

| 树 | 提交 | 内容 | 对应意见 |
|---|---|---|---|
| old | `5e4596dc7` | PR #4751 当前 head | |
| c1 | `a7864a5a3` | 手写反向，不再重算专家前向；fp32 梯度池只放 slot 行 | 1、2、9 |
| c2 | `06d8c338d` | 路由权重在 expert op 里乘；用 act_output 求权重梯度（K3 报告 §5.2.2） | 7 |
| c3 | `493ce26e0` | MoE 可选：共享专家放在 side stream（默认关） | 3、4 |
| c4 | `1db6fbcd1` | slot 梯度规约改为异步，在 MoonEP 的 stream 上和 dx GEMM 重叠 | 5 |

## 上机

1. 先查多播：`python3 -c "import ctypes;c=ctypes.CDLL('libcuda.so.1');c.cuInit(0);v=ctypes.c_int();c.cuDeviceGetAttribute(ctypes.byref(v),132,0);print('multicast',v.value)"`，要 1。
2. 环境：
   - 驱动是 560 加 compat（CUDA 12.8）：用本机 `/workspace/wheels_h100_68e0ae4/` 里的 torch wheel，配合 `kit_h100_2026-09-30/setup_venv.sh`。
   - 驱动支持 CUDA 13：直接装 0928 nightly。
   - MoonEP 装 `33327eb`，`nvidia-cutlass-dsl` 装 4.6.2（`kit_moonep_rewrite_2026-09-29/setup_box.sh`）。
   - 布局和 09-30 那台一样：`~/mep/venv_src`、`~/mep/tt`。
3. 把本 kit rsync 到 `~/kit/kit_moonep_perf_2026-10-03`，用 `setsid nohup bash h100_moonep_perf.sh > /dev/null 2>&1 & disown` 启动，然后边跑边拉 `~/mep/results/moonep_perf`。

## 预期

- **GPU 单测**：`test_moonep.py` 9 passed（2 卡 7 个、4 卡 2 个），共享专家 stream 的测试 1 passed。
- **逐位比对**（真 kernel）：
  - old 对 c1、c3 对 c4，6 个场景都应 BITWISE EQUAL。
  - old 对 c2：output 逐位相同；梯度只有 bf16 舍入级的差；两边对 fp32 参考的误差应在同一量级。
  - 5060 上假包的结果：old 对 c1、c3 对 c4 全部逐位相同；old 对 c2 的梯度差 2e-3 到 1e-2（相对最大值）。
- **microbench**：
  - expert op 的前向加反向，c1 应比 old 少一次前向 GEMM（约 1/4 的专家 GEMM）和一轮 fp32 拷贝。
  - c2 少一个保存的 fp32 输出。
  - c4 的 `reduce_grad` 被 dx GEMM 盖住；偏斜路由时 slot 多，差别应更明显。
  - MoE 层：c4 打开 stream 后，共享专家和路由分支重叠。
  - 峰值显存：c1 多存 gate/up，但梯度池减半。
- **端到端**（debug 模型）：
  - 第 1 步 loss：old 和 c4 应完全相同，因为前向没变。
  - grad norm 和后续步数：差在舍入级（c2 改了舍入顺序，c3 改了 x 梯度的累加顺序）。
  - 对照：标准 EP 跑两次（噪声底），以及 MoonEP 对标准 EP。

## 没做的

- **第 6 条（零拷贝）**：dispatch 是 ORDERED 的 op，SAC 会缓存它的输出，零拷贝的视图会被下一次 dispatch 覆盖，所以不能直接打开。K3 报告的做法是反向再 dispatch 一次来拿 x，加上零拷贝；这需要把三个 op 合成一个，留到后面。
- **第 8 条（本地行 alias 参数）**：需要 bf16 参数放在 VMM 内存里，是配方级的决定。
- **加宽的 debug 模型**：之前的约定是只加宽、不加深，并且要先问。这一轮用 microbench 测 K3 的单专家宽度，不加宽整个模型。
