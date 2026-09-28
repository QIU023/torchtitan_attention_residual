# 5060 冒烟（2026-09-27 夜到 09-28）

用户："先暂停h100了 你这么多pr分支都重新审核diff并且确保5060smoke完成"。给那些 head 还没在 GPU 上跑过的 PR 分支做冒烟，设置都是已有记录的，结果只进 logbook，不进 body。

- 机器：8 × RTX 5060 Ti 16 GB，torch 2.15.0.dev20260906（`/workspace/venv_bfx9`）。现在 main 上的树要打本地 torch 兼容补丁（`scratchpad/lbplan/pr5_torch_compat_shim.patch`），跑完撤掉。
- 脚本：`scripts/run.sh`（整个队列），`scripts/tab_a.py`（A 组的表），`scripts/tab_cd.sh`（C、D 组的表，调用 `kit_pp_lowerbound_2026-09-26/tab_lb.py` 和 `balance_table.py`）。
- 树：main `f35966713`，#4656 `aa6d9fedc`，#4881 `ae3a7881b`，PR A `439bd2088`，#4765 `61734f376`，#4764 `71e8bfaf2`。

| 组 | 设置（出处） | 格子 |
|---|---|---|
| A | 单卡，body 的 recipe：debug model，2048 token/步，512 一个 micro-batch，10 步，seed 42，deterministic，一份预热 cache | main 和 #4656 × none / selective / full；#4881 关 AC |
| B | B200 组合格 `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4`，8 卡，10 步（recipe 不固定 seed） | #4656、PR A |
| C | PR A 表的 PP 布局（`kit_pp_lowerbound_2026-09-26` 的 s6）：93 层、block 12、每 stage 3 层、dim 2048、seq 2048、M16、FullAC、pp8 × vp4，6 步，一份预热 cache | #4656、PR A、#4765 `cpu_offload=all`、#4764 `planned` |
| D | b1 的 balance 布局：同上但关 AC、dim 1024、seq 512，6 步，一份预热 cache | #4764 什么都不开、只开 balance（tcp）、planned 加 balance |

结果在 `results/`，汇总在 `DIFF_REVIEW_ALL_2026-09-27.md` 的最后一节。
