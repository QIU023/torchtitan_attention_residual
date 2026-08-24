# PP 数值结果(文本侧,for draft PR)

树:`k3_on_4025` @ 已 rebase 到 upstream/main。flavor `kimi_k3_debugmodel_text_32l`
(32 层,3 KDA : 1 MLA,无视觉塔;24 层无法整除 pp8 x vp4 的 32 段,故用 32)。
单一 seed checkpoint,每格自带同配置预热,每个计量格断言 `Loading the checkpoint`。

## pp x vp 全交叉,step-1 逐位相同

| cell | schedule | 段布局 | step-1 loss |
|---|---|---|---|
| dp1 | - | - | 12.45788 |
| pp2 | 1F1B | 2 段 | 12.45788 |
| pp4 | 1F1B | 4 段 | 12.45788 |
| pp2 x vp2 | Interleaved1F1B | 4 段 (2/rank) | 12.45788 |
| pp2 x vp4 | Interleaved1F1B | 8 段 (4/rank) | 12.45788 |
| pp4 x vp2 | Interleaved1F1B | 8 段 (2/rank) | 12.45788 |
| pp4 x vp4 | Interleaved1F1B | 16 段 (4/rank) | 12.45788 |
| pp8 x vp2 | Interleaved1F1B | 16 段 (2/rank) | 12.45788 |
| pp8 x vp4 | Interleaved1F1B | 32 段 (4/rank) | 12.45788 |

**九个 PP 配置的 step-1 loss 与 dp1 逐位相同。** PP(含 VP/交错调度)是 behaviour-free。

## 方法学

* 同一个 seed checkpoint 载入每一格 —— 跨格比较只有同 seed 才有意义。
* 每格先跑一次丢弃的同配置预热,再计量 —— 冷/热 inductor 缓存的差(~7e-3)大于
  真实跨格差,不预热会把它误读成并行 bug。
* 每个计量格断言日志出现 `Loading the checkpoint from`,否则标 ASSERT-FAIL ——
  静默 fresh-start(seed 没载入)是最隐蔽的污染源。
