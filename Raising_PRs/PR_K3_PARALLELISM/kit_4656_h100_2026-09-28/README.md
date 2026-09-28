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

## 第一张表改成 65536 token 的 micro-batch（09-28 07:39 到 07:48，用户："第一张表的 micro-batch 改大到 65536"）

- `run_4656_t1_65536.sh`，结果在 `results_t1/`：262144 token/步（4 个 65536 token 的 micro-batch），其余同原来的第一张表。三种 AC 都是 10/10 逐位相同；关 AC 52.12 → 37.94 GiB，tps 149269 → 134285（新 cache 150143）；selective 11.30 → 11.07、full 6.30 → 6.09 GiB，tps 106657 对 106692、103525 对 103927。
- 原来 512 token 那版第一张表（`results/` 里的 `id_*`）和 selective 补测（`results_sel/`）不再使用。

## #4656 head 变化后的核对（09-28）

- `check_head_f181.sh` → `results_f181/`：第一次收小的 `f181f3420` 对量数字的 `f14d681f4`，关 AC 和 selective 各 10 步，逐位相同，显存相同。
- `check_head_5d46.sh` → `results_5d46/`：第二次收小的 `5d469fdf3`（模型对外保留 stack，stage 不动）对 `f14d681f4`，同样逐位相同，显存 37.94、11.07 GiB，tps 差 1.5% 以内。
