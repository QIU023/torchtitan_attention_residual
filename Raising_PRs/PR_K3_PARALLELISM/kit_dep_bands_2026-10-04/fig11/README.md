# 图 11 逐格复现（10-04）

- 原图：K3 报告 PDF 第 19 页，`pdftoppm -r 500 -f 19 -l 19 -png <pdf> p19`，`pdftotext -f 19 -l 19 -bbox <pdf> p19_bbox.html`（两个文件太大，不进仓库）。
- `scan.py`：沿 PP0 到 PP2 三行扫出每个方框的起止（单位：一个文本前向）和颜色；`transcribe.py` 用 bbox 里的数字给方框编号，写出 `fig11.json`（156 个方框）。
- `order.py`：每行 48 个文本动作和 torch `ScheduleInterleaved1F1B`（pp3 × vp4，6 个 micro-batch）的顺序逐个相同。
- `sim_rhythm.py`：锁步 1F1B 节奏，144 个文本方框全部对上；`sim_grid.py`（规划器用的 torch 网格）、`sim.py`（只看依赖）、`sim_megatron*.py`（每步阻塞交换）是对照。
- `reproduce.py <worktree>`：规划器按图的比例（ViT 前向 0.5、反向 2 倍）跑，画到图的时间轴上，156 个方框和 `fig11.json` 逐个比；`draw.py` 生成 `fig11_vs_plan.png`（上：原图，下：规划器）。
- `check_plan.py`：我们自己的代价（ViT 反向 3 倍）下，代价比 0.01 到 1/3 排布和图一致，0.34 起不一致；`check_fig_costs.py`：图的比例下 0.5 一致、0.51 起不一致。
- `run_with_old.py <old plan.py>`：新测试拿到旧规划器上跑（`6f5312fab`、`6b580e438` 各失败 7 个 subtest）；`k25_same.py`：不开 bubble 时新旧规划结果逐项相同（592 组）。
- 运行：`PYTHONPATH=<wt_dep> /workspace/venv_0928/bin/python <script> <wt_dep>`，`<wt_dep>` 是 `dep_review1` = `5b01a6932` 的工作树。
