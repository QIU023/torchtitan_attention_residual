# TP PR 分支证据(2026-08-27)

判据见 `EVIDENCE_METHOD_2026-08-25.md`。

分支:`k3_tp` @ `c0345b915`(fork `QIU023/torchtitan`,基线 `upstream/main` 30eb5e502)。
矩阵**跑在分支 worktree** `/workspace/wt_k3_tp` 上(FILING.md 规则:数字属于
reviewer 看到的那份 diff),不是集成树。矩阵目录 `mx3_tp_branch_0827_222522`。

flavor `kimi_k3_debugmodel`(默认即多模态)。全局 batch 8192,微批 512/dp rank,
seq 512,seed 42,`--debug.deterministic`,单一 seed checkpoint,每格预热后读第二遍,
每格断言 `Loading the checkpoint from`。**6 格,0 挂**。

## 一、分支构成(与 k3_ep 同模式:上游 base + 小增量,三 commits)

| commit | 内容 |
|---|---|
| e27367ed4 | distributed/utils.py:grad-norm 按参数 mesh 分组(+43/-3) |
| f4e78b30a | [kimi k3] enable tensor parallel:sharding.py +169(声明),dtensor_ops.py +41,kda.py 内核调用点 unwrap,model.py +19(接线 + splice lift),parallelize.py 去门 |
| c0345b915 | 多模态模型测试 fsdp2 -> fsdp2 x tp2(4 卡) |

与集成树的差异(刻意):无 ep+tp 内联 override(EP 未上游)、无 CP 侧
unwrap(conv_with_halo/Ulysses 不存在于此分支)、不收编 EP 裁剪旁路(避免 TP PR
越界改 EP 行为,只给主路径加 mesh 分组)。声明入 `sharding.py`(qwen3_5/k3_ep
同模式),TP-only 版本 `set_tensor_parallel_sharding_config` 无 enable_ep 参数。

## 二、表(TP 单轴 + TP x DP,不含其它轴)

| cell | world | step 1 | step 2 | step 3 | step 10 |
|---|---|---|---|---|---|
| dp1 | 1 | 12.60370 | 9.67929 | 7.40097 | 3.38761 |
| tp2 | 2 | 12.61424 | 9.55940 | 7.14961 | 3.30040 |
| tp4 | 4 | 12.61019 | 9.50305 | 7.13944 | 3.32639 |
| tp8 | 8 | 12.60325 | 9.58821 | 7.14432 | 3.37905 |
| dp2 | 2 | 12.57909 | 9.98178 | 7.55822 | 3.42322 |
| fsdp2 x tp2 | 4 | 12.59098 | 9.87587 | 7.62463 | 3.42794 |

## 三、判读

- **step-2 判据成立**:|tp2-dp1| = 1.20e-1,|tp4-dp1| = 1.76e-1,|tp8-dp1| =
  9.1e-2;而 |dp2-dp1| = 3.02e-1 —— 只改 DP 度带来的移动是头切分的 ~2.5 倍。
  mesh 格 |fsdp2_tp2 - dp2| = 1.06e-1,同量级。与 CP/EP/PP 正文的论证同构。
- **tp 三格聚簇**(s3 7.139-7.150):TP 的归约顺序移位方向一致、跨度数稳定,
  无离群格,无需重跑仲裁。
- s1 前向即有 ~1e-2 差:`wo` rowwise 切 k 维改变求和顺序,bf16 下相对差
  ~8e-4,与 CP 表的 s1 散布同量级,家族正常。
- warm↔measure 在 s10 已差 0.14(dp1):冷 inductor 缓存的已知机理,正是
  只读第二遍的原因;分支表不声称"每格重跑逐位一致"(未做第三遍验证)。

## 四、状态

PR 正文 `Raising_PRs/PR_K3_PARALLELISM/PR_BODY_TP.md` 已填数;分支已推
`origin/k3_tp`。**未经用户逐字确认不提交 PR。**
