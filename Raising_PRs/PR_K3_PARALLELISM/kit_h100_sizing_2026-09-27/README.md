# H100 尺寸的推算（2026-09-28 凌晨，5060）

给 `H100_SIZES_2026-09-27.md` 第 3、4、5 节定宽度用；结果只用来定尺寸，不进 body。

- `count_params.py`：在 meta 上数 PP 探针（`kit_pp_lowerbound_2026-09-26/probe_lb.py`）每个 rank 的参数，切分用 titan 自己的 `_get_pipeline_metadata` 和 `_generate_llm_fqn_per_model_part`，stage s 在 rank s % pp。输出 `count_params_main_pp4.out.txt`。
- `sizing.sh` + `run_lb4_5060.sh`：5060 上 pp4 × vp8（93 层、block 12、每 stage 3 层、M16、seed 42）的四格，各 4 步：
  - FullAC、seq 2048，main `f35966713`，dim 1024 和 1536（`f_d1024`、`f_d1536`）；
  - 关 AC、seq 512，#4764 `71e8bfaf2` 什么都不开，dim 768 和 1024（`n_d768`、`n_d1024`）。
  - 本地 torch 兼容补丁跑前打上、跑完撤掉（`results/worktrees_after.txt`）。
- `fit_width.py`：每个 rank 的峰值 = 16 B × 该 rank 的参数（meta）+ a + b × dim，a、b 由两格实测拟合；输出 `fit_fullac.out.txt`、`fit_noac.out.txt`。
- `dep_ratio.py`：视觉塔参数和文本每层激活参数（路由专家按 top_k / num_experts 计），输出 `dep_ratio.out.txt`。

| 布局 | 5060 实测（第 3 步，每个 rank，GiB） | 推算 dim 7168（GiB） |
|---|---|---|
| FullAC、seq 2048，main | dim 1024：7.95 / 8.69 / 8.68 / 8.48；dim 1536：11.64 / 12.58 / 12.57 / 12.30 | 55.3 / 55.5 / 55.4 / 54.3 |
| 关 AC、seq 512，#4764 都不开 | dim 768：10.00 / 10.20 / 9.76 / 10.18；dim 1024：11.69 / 11.91 / 11.40 / 11.79 | 55.7 / 53.0 / 50.7 / 50.5 |

外推跨度是 4.7 倍（FullAC）和 7 倍（关 AC），上机第一步用 2 步实测核对。
