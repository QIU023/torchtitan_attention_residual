# #4656 的两张表，单卡 H100（2026-09-28）

用户："body改掉，标注debugmodel，tokens per micro-batch 这个往上改，直到main OOM；决定了 单卡H100"。

- `run_4656_h100.sh`：一张 H100，一格一格顺序跑。main `f35966713`（`~/w/main`）对本 PR `f14d681f4`（`~/w/c4780`，checkpoint 加列表载体），main 上的 `kimi_k3_debugmodel` 原样。
  - (a) 第一张表：2048 token/步，512 一个 micro-batch，10 步，none / selective / full，一份预热 cache，每格在自己的拷贝上量；另加 main 关 AC 在自己的新 cache 上的一行。
  - (b) 第二张表：关 AC，一个 micro-batch，3 步，每个配置自己预热一步再在拷贝上量；tokens per micro-batch 取 512、4096、8192、16384、32768、65536、98304、131072，main 第一次 OOM 就停。
- `tab_4656.py <results dir>`：输出 body 的两张表，替换 `PR_BODY_4656_v4_2026-09-28.md` 粘贴区里的两处 `Pending (H100).`。
- 机器上要有：`~/venv_pp`（torch nightly、attn-gym、torch_remat、spmd-types、cutlass），`~/w/main` 和 `~/w/c4780` 两个 detached worktree。新机器按 `kit_h100_2026-09-27/scripts/setup_pp.sh` 装。

## 结果（09-28 06:50 到 07:26，`135.135.24.114`，1 × H100 80GB，torch 2.15.0.dev20260926+cu130）

- `results/`：`run_4656_h100.sh` 的每一格（`run.log`、`rc`），`trees.txt`（main `f3596671`、本 PR `f14d681f`，都干净），`progress.txt`。
- `results_sel/`：`rerun_selective.sh`，selective 下两棵树各再跑两遍：main 1154、1127，本 PR 1158、1115 tps，第一张表里 1110 对 1160 是重复跑的波动。
- 表：`python3 tab_4656.py results`，已填进 `PR_BODY_4656_v4_2026-09-28.md`。
- main 在 98304 OOM（`RuntimeError: Triton Error [CUDA]: out of memory`，原脚本的 OOM 检测没认出这种写法，已改成不分大小写匹配 "out of memory"），131072 也 OOM；本 PR 98304 是 56.67 GiB，131072 是 75.51 GiB（不进表，表停在 main 第一次 OOM）。
