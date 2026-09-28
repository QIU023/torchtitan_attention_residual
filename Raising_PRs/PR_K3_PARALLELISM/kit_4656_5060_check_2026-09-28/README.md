# #4656（`5d469fdf3`）在 TP 加 SP、PP 下对 main 的逐位核对（2026-09-28，8 × 5060）

用户："按第 3 条补跑 mm 格，确认 sharding 的改动在 TP 加 SP 下没问题"。main `f35966713` 对 #4656 `5d469fdf3`，seed 42，deterministic，每个格子一份预热 cache，10 步；本地 torch 兼容补丁跑前打上、跑完撤掉。

| 格子 | 结果 |
|---|---|
| `kimi_k3_debugmodel_mm`（fsdp2 × tp2 开 SP × ep2，typecheck 开，4 卡），`run.sh` | main 和 #4656 都在第一步报同一个错：`SpmdTypeError: SPMD type mismatch on axis mesh_ep: tensor has P, expected I`（`protocols/module.py` 第 577 行的边界重分布）。main 自己在这个环境就过不了，和 #4656 无关；spmd-types 是 main 固定的 0.2.5 |
| `kimi_k3_debugmodel_fsdp2_tp2_ep2_pp2_vpp4`（8 卡，PP recipe 本来就关 typecheck），`run.sh` | 10 步 loss 和 grad norm 逐位相同（第 10 步 3.33909 / 2.4531）；debug model 带视觉塔，layer 0 的多模态边界在这一格里用到 |
| 同一个 mm 格关掉 typecheck（本地 recipe `mm_notc.py`），`run2.sh` | 10 步 loss 和 grad norm 逐位相同（第 10 步 3.65948 / 3.1406）；日志里的显存 1.06 → 0.83 GiB |

每格的每步记录在 `results/res/`（第一轮）和 `results/res2/`（关 typecheck 的 mm）。
