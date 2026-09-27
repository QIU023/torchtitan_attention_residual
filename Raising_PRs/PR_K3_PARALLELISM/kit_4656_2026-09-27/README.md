# #4656（`attnres_review1` `f14d681f4`）的 5060 实测，2026-09-27

8 × RTX 5060 Ti，每格一张卡，debug model，bf16，seed 42，deterministic，本地 torch 兼容补丁（`../kit_pp_review5_rebase_2026-09-26/pr5_torch_compat_shim.patch`，不进任何提交）。main 是 `f35966713`（临时 worktree `wt_main_f359`），PR 是 `f14d681f4`（`wt_attnres`）。每个目录有 `run.log`（rank 0 的完整日志）和 `rc`。

| 目录 | 内容 | 用在哪 |
|---|---|---|
| `warm_{main,pr}_{none,selective,full}` | 1 步，预热共享的 cache0 | |
| `id_{main,pr}_{none,selective,full}` | 10 步，每格用 cache0 的一份拷贝，2048 token/步，512/micro-batch | body 的主表：loss 和 grad norm 10/10 相同；显存取日志里的 max reserved；tps 取第 6 到 10 步的平均 |
| `id_main_none_fresh` | main、AC 关，用自己的新 cache | 噪声行：cache 本身让 tps 差 4% 左右；显存里混了 autotune，不可比 |
| `sc_{main,pr}_{2048,4096,8192}` | 新 cache 上的 token 扫描（3 步，一个 micro-batch） | 作废：reserved 里混了 autotune |
| `swwarm_*`、`sw_*` | 同一扫描，每个配置先跑 1 步预热自己的 cache，再在拷贝上跑 3 步 | body 的扫描表：第 3 步 1.89 → 1.42、3.51 → 2.62、6.71 → 4.95 GiB，loss 3/3 相同；tps 不稳（第 3 步读数跳到几千），不报 |
| `compose_4656`、`compose_pra` | B200 组合格 `kimi_k3_fsdp2_tp2_ep2_pp2_vpp4`，8 卡 10 步，无固定 seed | 冒烟：两棵树都 rc=0（#4656 8.14450 → 3.47590，PR A `e8d0a4aec` 8.08985 → 3.49180） |

脚本：`run_acr.sh`（预热、一致性格、噪声行、新 cache 扫描），`run_sweep_warm.sh`（预热后的扫描），`queue_s6_and_smokes.sh`（PR A 对 #4656 的 campaign s6 加两个冒烟格；s6 的结果在 `../kit_pp_lowerbound_2026-09-26/results/s6_*`）。
