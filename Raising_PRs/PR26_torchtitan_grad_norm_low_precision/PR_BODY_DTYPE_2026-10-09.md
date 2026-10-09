# titan：`clip_grad_norm_` 传 `dtype=torch.float32`（接替 #4135），2026-10-09

## 状态（不粘贴）

- **10-09 用户定稿（"不需要跑数值，4135都审过了，把PR开了 ... 就改这三行；body里面就说follow #4135 conclusion, upstream PR merged 之类的就行，标题得写 pending titan release bind to torch 2.16"）：** 不跑数值，下面"GPU 会话待跑"一节作废。粘贴区换成短版：Summary 一句加一条 bullet，再加一句说明在等绑定 torch 2.16 的 titan release；标题加上 "[Pending torchtitan release bound to torch 2.16]"。分支不变，仍是 `grad_norm_fp32_dtype` = `9d60f8c6d`，只有三行。

- **分支：** fork `QIU023/torchtitan` 的 `grad_norm_fp32_dtype` = `9d60f8c6d`，基于 upstream main `31b503c89`，一个提交，只改 `torchtitan/distributed/utils.py` 三行（第 549、608、615 行的 `get_total_norm` 调用各加 `dtype=torch.float32`）。
  - 本机检查：black 22.12、pyflakes 干净。
  - 提交信息：作者 QIU023，没有 trailer，也没有跨仓库引用（写作 PR-194033）。
- **titan 怎么绑定 torch：**
  - main 跟 nightly：pyproject 不声明 torch；README 写源码安装 "requires the nightly build of PyTorch"；CI 的 matrix（`set-matrix.yaml`）是 `index-url .../whl/nightly/cu132`，`torch-version` 为空，每个 job 都装当时最新的 nightly，ROCm 也是 nightly；代码里没有任何 torch 版本判断。
  - 2026-10-08 的 nightly（源提交 `79ef85d9b6`，版本 2.16.0.dev）已包含 pytorch 的 `294bc517d9a7`（compare：领先 47，落后 0），所以现在开 PR，CI 就能用上 `dtype`。
  - 只有 titan 自己的 release 分支才钉 stable torch：`docs/release.md` 规定每个 PyTorch 小版本切一个 titan release 分支，切之前用该版本的 RC 验证 main。
- **时间点上的风险（要告诉 maintainer）：**
  - pytorch `release/2.15` 是 10-05 从 main 切出的（merge-base `2cce586c7744`），没有 `dtype`（release/2.15 的 `clip_grad.py` 里查不到这个参数）；它 10-07 才合入 main，所以只进 2.16。
  - titan 对应 torch 2.15 的 release 分支（应是 `release/0.4`）还没切。上一轮是 pytorch 08-11 切 release/2.14，titan 08-19 切 release/0.3，相隔 8 天。照这个节奏，这一轮大约在 10-13 前后。
  - 如果本 PR 在那之前合进 main，titan 用 torch 2.15 RC 验证 main 时，`clip_grad_norm_` 会报 `TypeError`（不认识 `dtype`），每个训练都会挂。
  - 我的建议：PR 现在就开（ready，不要 draft，#5145 的教训），body 里写明需要 2.16 或 10-08 以后的 nightly，合入时机让 maintainer 定。不加版本判断：titan 代码里没有先例，到 2.16 以后也是死代码。
- **GPU 会话待跑（Results 和 Test plan 的数等这些填）：**
  1. 环境：torch nightly ≥ 2.16.0.dev20261008。
  2. 单测：`pytest tests/unit_tests/cpu/test_trainer.py tests/unit_tests/cpu/test_invalid_loss.py -q`，以及 4 卡 `tests/unit_tests/gpu/test_fsdp_moe_sharding.py`（EP 路径）。
  3. 数值表：`llama3_debugmodel`，`--training.dtype bfloat16`，dp_shard 2 固定，pp 取 1 / 2 / 4。PP 不改变 loader 的分片，所以三行读的是同一条数据流。main 和本 PR 共用同一份预热过的 cache，seed 固定、deterministic，报 step 1 的 loss 和 grad norm：main 下三种 pp 的 grad norm 应各不相同，本 PR 应三者相同。
     - 再加 float32 的两行（main 对本 PR，应逐位相同），以及 main 跑两次的噪声底。
     - step 10 和 20 先看 main 的轨迹，按数值表规则决定报到哪一步。
  4. EP 路径：任意一个 EP=2 的格子跑 10 步，确认能跑通、grad norm 是 fp32。
- 粘贴区已检查：没有 we/our/us、破折号或非 ASCII 字符。

标题：`[Pending torchtitan release bound to torch 2.16] clip_grad_norm_: accumulate the total norm in float32`

--- PR body: PASTE BEGIN ---

## Summary

Follows the conclusion of #4135: the `dtype` argument for `torch.nn.utils.get_total_norm` went into PyTorch instead of a copy in titan, and pytorch/pytorch#194033 has merged (PyTorch 2.16, and nightlies from 2026-10-08).

- `torchtitan/distributed/utils.py`: the three `get_total_norm` calls in `clip_grad_norm_` and `_clip_grad_norm_with_ep` pass `dtype=torch.float32`, so with bf16 gradients the total norm accumulates in float32 and no longer depends on the PP or EP split.

PyTorch 2.15 does not have the argument, so this is pending the titan release that binds to torch 2.16.

--- PASTE END ---
