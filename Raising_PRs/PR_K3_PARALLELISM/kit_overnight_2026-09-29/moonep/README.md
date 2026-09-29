# MoonEP 负载均衡的证据方案（2026-09-29 夜，T3b）

目的：#4751 body 说 MoonEP "keeps every rank's routed token count at S x K by prefetching copies of hot experts"。要证明这句话，得有每个 rank 的负载统计和步时，loss 只能说明没算错。今晚只准备文件、在 CPU 上核对，GPU 由主会话排队。

## 文件

- `moonep_load.py`：本地 recipe 和探针，不提交。
  - `load_std` / `load_moonep`：Kimi K3 debug model（dim 256、17 层）换成 `LOAD_E` 个专家（默认 128）、top-`LOAD_K`（默认 8）、2 个 shared，FSDP4 × EP4，seq 512，AdamW(lr=8e-4)。换 AdamW 是因为 recipe 里的 DistMuon 按原来 8 个专家的模型配好了切分，换模型以后对不上。
  - `LOAD_SKEW`：在选专家用的分数上加固定偏置 `-LOAD_SKEW * log(1 + e)`（不改 gating 值），编号小的专家热，它们都在 EP rank 0 上。0 是自然路由。K3 的 router 是 `QuantileBalancedTopKRouter`，补丁打在它的 `_select_experts` 上。
  - `LOAD_NO_BIAS=1`：关掉 K3 的 quantile balancing hook（`register_moe_quantile_balancing_hook`），让偏置在一次运行里不被平掉。
  - 探针：每个 MoE 层每个 micro-batch，用 router 的每专家计数在 EP 组上求和，再按每个 rank 的本家专家加起来，得到静态放置下每个 rank 会收到多少 token，记 max / mean。MoonEP 上再逐次 dispatch 核对：这个 rank 的真实行数（补齐后的组末尾减去补零行）是否正好 S × K，以及有几个槽放了副本。反向重算时不记。
- `run_load.sh`：先各预热一步（同一份缓存），再跑数值和负载格：standard、MoonEP × 自然路由、偏置，各 20 步，seed 42，确定性。`MODE=real` 时再跑计时格（不开确定性，30 步）。
- `tab_load.py`：出表。

CPU 上核对过：两个 recipe 都能建出来（16 个 MoE 层，128 个专家，top-8，standard 走 `AllToAllTokenDispatcher`，MoonEP 走 `MoonEPRoutedExperts` 加 `MoonEPTokenDispatcher`），偏置补丁和 hook 开关都生效。

## 命令

**(a) 假 MoonEP，本机 4 × 5060**（只验证记账和流程；假包的路由和计时没有意义；假包的 `Buffer` 已补上公开版的 `explicitly_destroy` 等参数）：

```bash
S=/tmp/claude-0/-workspace/55727fa0-a690-442c-a59f-5ed87d136f52/scratchpad
K=/workspace/torchtitan_attention_residual/Raising_PRs/PR_K3_PARALLELISM/kit_overnight_2026-09-29/moonep
# 树：moonep_review1 = 1633dcd79 的 worktree（fork 上已推；本机 $S/wt_moonep_rb 就在这个提交上）
MODE=fake TREE=$S/wt_moonep_rb VENV=/workspace/venv_0928 OUT=$S/moonep_load GPUS=0,1,2,3 bash $K/run_load.sh
cat $S/moonep_load/tables.md
```

预计 10 到 15 分钟（两个预热各含一次冷编译，四个 20 步格）。

**(b) 真 MoonEP，4 × H100（NVSwitch，multicast 为 1）**，环境照 `kit_h100_2026-09-29/setup_src.sh`（MoonEP 在 `~/mep/MoonEP_src`）：

```bash
git -C ~/mep/tt fetch origin moonep_review1 && git -C ~/mep/w/moonep checkout -q --detach 1633dcd79
# 本目录拷到机器上 ~/kit/moonep_load
MODE=real TREE=~/mep/w/moonep VENV=~/mep/venv_src OUT=~/mep/results/moonep_load bash ~/kit/moonep_load/run_load.sh
```

预计约 15 分钟（预热 2 × 1 分钟，数值 4 × 1 分钟，计时 4 × 1.5 分钟）。另外 `kit_moonep_rewrite_2026-09-29/smoke.sh` 的 GPU 单测和 CI 格要在新 head 上再跑一次（约 10 分钟），新 head 加了 reduce 前的栅栏。

## 判断标准

- 自然路由：standard 的 max / mean 记下来（128 个专家、top-8，刚初始化时预计在 1 到 1.5 之间）；MoonEP 每次 dispatch 都应当正好 S × K 行。
- 偏置：standard 的 max / mean 应当明显大于 1（rank 0 的专家热）；MoonEP 仍然每次正好 S × K，槽里有副本；计时格里 MoonEP 对 standard 的步时差就是 body 要的收益。
- 数值：standard 和 MoonEP 第 1 步 loss 逐位相同（本机假包上是这样），之后的差应当和噪声底同一量级。
- 真 MoonEP 上还要看 reduce 栅栏（`1633dcd79`）以后，GPU 单测里专家权重梯度是否仍然和稠密参考一致。
