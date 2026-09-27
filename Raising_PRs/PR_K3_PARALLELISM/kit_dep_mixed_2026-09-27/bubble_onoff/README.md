# DEP bubble 开和关（新 head `31f372593`，2026-09-27 晚，4 × RTX 5060 Ti）

- 配置：B200 套件的 `kimi_k3_debugmodel_pp4_vp2_vit_dep`，本地 recipe `dep_bubble.py` 只把 `vision_dep.bubble` 关掉（不进分支）；seed 42，deterministic，10 步；本地 torch 兼容补丁在跑时打上、跑完撤掉。
- cache：`warm_on`、`warm_off` 各跑 1 步预热同一个 cache0，之后每格用一份拷贝。
- `on` / `on2`、`off` / `off2`：每个设置各跑两遍，各自 10 步逐位一致；开和关第 1、2 步相同，第 3 步起分开。
- `probe_on` / `probe_off`：`dep_bubble_probe.py` 在每次 optimizer step 之前 dump 每个参数的梯度，跑 3 步；`cmp_grads.txt` 是逐参数比较：第 1 步全部相同；第 2 步（第一次放置）四个 rank 上 443 个文本参数相同，22 个视觉塔参数全不同（相对差 1.2e-3 到 3.1e-3，梯度 bf16）；第 3 步全部不同。
- 第 1 步按设计全部 inline 编码（`vision_dep.py`：FSDP 第一次 root forward 之前编码会让视觉塔变成 root），放置从第 2 步开始。
