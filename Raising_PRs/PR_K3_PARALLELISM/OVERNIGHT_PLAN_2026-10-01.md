# Overnight 10-01（5060 only；H100 已停）

用户 10-01："H100停了 overnight做这个，整理一下overnight goal，然后MoonEP维持现状，另外本机有另一个claude agent在做Research实验（OPRD+SATS）……告诉它你什么时候验证完成5060 DEP比例问题，然后你自己commit push logbook，明天我们再去H100验证DEP；它会接管本机5060以smoke实验代码"。另附 CPU 会话的审核意见（`DEP_FIX_PLAN_REVIEW_2026-10-01.md`）。

## 目标

1. **DEP 代价比例（5060）：** 把负载的代价比例放进 K3 的区间（0.1 到 0.3），看规划器在这个区间里怎么放，实际藏住多少。规划器不改（审核建议，用户同意）。
   - 代价比例 = 一个带图 micro-batch 的编码前向时间 / 16 stage 切分（pp4 × vpp4）的中间 stage 前向时间，这是规划器的 `vision_dep.bubble_cost_ratio`（`DEPV_COST_RATIO`）。
   - 在 5060 上实测这个时间比，挑出 0.1 到 0.3 的几档；
   - pp4 × vpp4、M16 跑 DEP 关 / K2.5 / bubble，记规划日志的放置（进气泡的编码、反向个数）和步时；
   - 一档开 profiler，用 trace 量 bubble 实际藏住的视觉时间；
   - 对照审核里的规划器模型表（代价比例 ≤ 1 时其余 12 次编码全进气泡；0.3 时反向 12 个，0.1 时 15 个）。
   - 5060 是 PCIe、没有 P2P，步时只作参考；比例和放置以这里实测为准，明天在 H100 上重新标定。
2. **明天 H100 的准备：** 根据 5060 的结果和 H100 09-30 的标定，定出 H100 上落在 0.1 到 0.3 的档位和脚本。
3. **MoonEP 不动**（`k3_moonep_seam` = `moonep_review1` = `e30d82886`）。
4. **收尾：** logbook commit / push；通知本机的 SATS-OPRD 会话 5060 可以用了（它在等我的消息，已确认不开 GPU 作业）。

## 已知的修正（开始前读代码发现）

- 审核里说 H100 那轮"塔跟着文本一起放宽到同样宽"，和 09-30 的脚本不符：`run_dep_h100.sh` 和 `calib.sh` 都是 `DEPV_TOWER=k3`，用的是发布版 K3 的塔（27 层、1024 宽），文本是放宽到 6144 的 17 层 debug model。
- 代价比例高的原因在时间而不在 FLOPs：H100 标定里 K3 塔对 224、448、768 px 的编码前向都是约 21 ms（1024 px 32.6 ms），27 层小矩阵乘的固定开销占主导；16 stage 切 17 层，中间 stage 前向只有 9.02 ms（seq 2048）。所以在 pp4 × vpp4、K3 塔下，图再小代价比例也在 2.3 以上，seq 8192 也约 0.6。
- 所以今晚 5060 上用不随文本放宽的 debug 塔（`DEPV_TOWER=base`：256 宽、2 层，`dep_ratio_local.py` 新加），配合序列长度和图的大小张数来定代价比例。规划器只看代价比例，塔的结构不影响放置；明天 H100 上是否保留 K3 塔，由用户在下面的选项里定。

## 步骤

- 阶段 1（`kit_overnight_2026-10-01/phase1_microbench.sh`）：单卡 microbench，dim 2048，`base` 塔 seq 2048 / 4096 / 8192，`k3` 塔 seq 2048 作对照。
- 阶段 2：按实测选 3 档（约 0.1、0.2、0.3），每档先 2 步显存冒烟，再 DEP 关 / K2.5 / bubble 各 10 步（同一份暖 cache），记规划日志和步时；一档加 profiler trace。
- 阶段 3：整理结果到 `DEP_RATIO_5060_2026-10-01.md`，写明天 H100 的档位和脚本。
