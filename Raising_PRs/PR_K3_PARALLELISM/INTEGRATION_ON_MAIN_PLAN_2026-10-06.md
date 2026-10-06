# 集成树搬到当前 main：进度和 Elfie 分支方案（2026-10-06，暂停）

用户 10-06："把最新的MoonEP和DEP和PR A，4764/4765 全部同步到rebase main后的集成树里面"。做到一半，用户改了方案（以 Elfie 的分支为底），随后让先停下，记录方案，先处理 MoonEP 的新问题和 #4380。

## 已经做完的（未接到任何 PR 分支上）

- 集成树 `k3_on_4025` 没动，仍是 `e5e47b857`（tag `k3_int_20260922e`），base 是 09-20 的 main `63c4e9fef`，在 #4810（模型配置和 parallelize 归属重构）之前，落后 main 169 个提交。
- 新树在 scratchpad worktree `wt_int1006`（detached），底是 main `e5c55525e`（10-05）。fork 上推了两个 tag：
  - `k3_int_20261006a` = `1120f99c5`：五条线 + #4639；
  - `k3_int_20261006b` = `5454b0ce8`：再加 optimizer 放宽。

| 新提交 | 来源 | 冲突和处理 |
|---|---|---|
| `6f3cc0a73` | #5025 `fc6302123`（Shuhua，未合）：AttnRes stage 打开 FSDP 手动 finalization | 无。torch dev20261003 起 FSDP × PP 必须有它 |
| `e6fa61f55` | #4656 `7dea4cbaf`（list，layer 平铺返回 blocks） | `model.py`：main 的 #4897 attention metadata 和 `recompute_needs_tensor` 都保留 |
| `7b9bd126c` | remat `e66a9442b`（PR A 栈里的 #4656 副本上面那个） | import：`FullAC/RegionAC/SelectiveAC` 和 main 的 `apply_local_compile` 都保留 |
| `577d053a7` | PR A `312bc8144` | 无。PR A 原来的 #4656 副本 `07abef619` 返回 list，这里底下换成了 `7dea4cbaf` 的平铺返回 |
| `b2442402e` | #4765 `fb5d35b56`（review 分支，10-01 CPU 会话叠的） | `stage.py`：#4661 的 `_forward_chunk_states` + #5025 那一行 + #4765 的 saves 上下文；`model.py`：`pp_memory` 放在 main 的 `local_compile_regions` 旁边，丢掉旧 base 的 `update_from_config`（main 已挪到 `set_sharding_`） |
| `f8265352c` `d49e0893d` | #4764 `50ef4dd1d` `908db8609` | 无 |
| `a872acbf2` | DEP `db050eb3f..6cda7daca` 作为一个改动 | `vision_dep` 和 `local_compile_regions`、`pp_memory` 并列；forward 加 `vision_embeds`，返回 PR A 的 list；入口先注册 routing 和 activation storage，再装 vision runtime |
| `b8d0025f7` | MoonEP `db050eb3f..6e3de1b8d` 作为一个改动 | `pyproject.toml` 保留 main 的 `dist_moe` 行；shared experts 的加法两条路径都保留 main 的 `recompute_needs_tensor` |
| `922839f01` | 集成修正 | remat 测试改用 `build_model_config`、kernel 的 `attention_metadata` 参数、平铺返回 |
| `b4e6e7988` | 集成修正 | DEP runtime 测试的玩具 stage 改成 block list（block 0 用 `hidden.view_as(hidden)`，对应原来的 `unsqueeze` 视图；用同一个张量会让单卡参考的梯度累加顺序变，差 1 ulp），routing 用生产入口的 `wait_sends_at_backward` |
| `1120f99c5` | #4639 `e5c55525e..ba86e5591`（acisseJZhong，K3 KDA CP）作为一个改动 | layer 0 边界保留 #4656 的 list（没有 stack 要转，不要 `enable_sp`）；embedding 同时接 DEP 的 `vision_embeds` 和 CP 的 `vision_bank_indices_T` |
| `5454b0ce8` | 集成树 `905ec0989` `acc8c6811` 按 main 的 `components/optim` 重写 | 整个冻结的 model part 不建 optimizer；空参数组初始化；scheduler 容器接受空 |

检查（venv_1003b，torch 2.15.0.dev20261003）：
- 在 `1120f99c5` 上跑 20 个 CPU 测试文件（五条线、DEP、MoonEP、CP、transforms、moe、rope、varlen、qwen3_5、integration definitions）：273 passed，1 skipped，0 failed。
- `5454b0ce8`：optimizer 和 LR scheduler 三个测试文件 45 passed。
- GPU 格子一个都没跑。

## 用户的新方案：以 Elfie 的分支为底

分支 `elfiegg/torchtitan` `work/kimi-k3-mx-qat-distmuon` = `b7b5eb2b9`（09-30）：main `9b8b9d244`（09-28，落后现在的 main 102 个提交）上的 9 个提交，+5354/−33，49 个文件。fork clone 里加了 remote `elfie`。

步骤（用户原话："拷贝checkout之后，rebase main然后把我们独有的全部cherry pick过去"）：
1. 比对两边各自缺少的 feature（main + 集成树 + 五条线，对 Elfie 分支）。
2. 拷出 Elfie 分支，rebase 到当前 main。
3. 把我们独有的全部 cherry-pick 上去：先是 `wt_int1006` 的 13 个提交，再是集成树独有、Elfie 和 main 都没有的部分。

### 比对（只读了 diffstat、提交说明和 Elfie 的 K3 README，代码还没逐行对）

Elfie 有、我们没有：
- packed MXFP4 的 HF 存储读取（`components/checkpointer/hf_storage.py`），K3 adapter 直接读 released 的 packed 专家；
- MX QAT（`quantization/mx_qat/{experts,linear,checkpoint}.py`、`config/transform/mx_qat.py`、`kimi_k3/quantization.py` 的 MXFP4 policy，以及 recipe）。它做在 main 的 GroupedLinear 上，QAT 选哪些模块由 checkpoint 里实际的 packed/scale 对决定；
- released checkpoint 形状规整（例如 96 头 KDA 的 `A_log` 存成 128 元素），以及 preflight 脚本 `validate_kimi_k3_mxfp4_checkpoint.py`；
- 经 FSDP 输出投影的 linear cross entropy（`loss.py`、`linear.py`、`training_engine.py`）；
- checkpoint 转换时 vision QKV 保持 replicated；
- 2.78T 验证脚本 `scripts/validation/kimi_k3/`（DistMuon、EP placement、pipeline 内存和启动、step watchdog、finite diagnostics）。

我们有、Elfie 没有：
- `wt_int1006` 上的全部：五条线、#4656 list、remat、#5025、#4639、optimizer 放宽；
- 集成树独有的：LoRA/QLoRA（packed MXFP4 冻结底座、merge/export、`quantize_lora_dcp.py`）、MTP、to_hf 两个修正（layer-0 占位放在持有 layer 0 的 stage；跳过 LoRA adapter）、rl flavors（veRL 用）、fsdp 的零值依赖、neighbor transport、ViT 动态 CP #4380；
- main 在她的 base 之后的 102 个提交。

两边重叠、要先逐行比的：
- QAT：集成树 `2cf71670e` 做在已删除的 GroupedExperts 上，Elfie 的在 GroupedLinear 上，预计取 Elfie 的；
- released 格式：集成树 `5b001bde1`（released layout 对照）和 `0073d4e65`（raster patch 布局），对照 Elfie 的 adapter 和形状规整；
- to_hf：集成树 `4fd7388bc` `11ac07691`，对照 Elfie 的 adapter 改动。

集成树里已被 main 或五条线取代、不再搬的：老 PP（main 已合 #4312）、老 AC reuse/RegionAC/remat、老 DEP、老 MoonEP、老 cache offload 和 pp_balance（换成 #4765/#4764）、per-head DistMuon recipe（main 有 #4596）、按 block 的 compile（main 改成 local compile regions）、attn-gym pin、老 CP 栈（换成 #4639）。

Elfie 分支 rebase 到 main 预计的冲突：
- `kimi_k3/config_registry.py` 已被 #4913 删掉，recipe 要挪到 `torchtitan_recipes/tests/models/kimi_k3.py`；
- `components/loss.py`（#5082 等 4 个提交）、`training_engine.py`（11 个）、`config/transform/{__init__,README,lora,quantization}.py`、`models/common/linear.py`、`tests/unit_tests/cpu/test_state_dict_adapter.py`。
- `kimi_k3/state_dict_adapter.py` main 之后没动过。

## 暂停时的顺序（用户 10-06）

1. 记录这个方案，logbook 拉取、rebase、推送。
2. 检查 MoonEP 新发现的问题。
3. maintainer 合了 CP（#4639），#4380 Dynamic CP 不再被挡住，先准备它。
4. 集成树按上面的方案再继续，开始前先问用户。

## 10-06 晚（overnight 目标："完成刚刚的集成树移植 在5060完整测试"）

新树在 scratchpad worktree `wt_int_elfie`（detached），底是 main `3f087cf15`（#4639 已合）。按用户的方案以 Elfie 的分支为底，自下而上：

| 提交 | 内容 | 说明 |
|---|---|---|
| `d04ddb1ed` | Elfie `9b8b9d244..b7b5eb2b9` 作为一个改动，作者记为她 | 她的提交中间状态引用已删除的 GroupedExperts（第 6 个提交才改成 GroupedLinear），逐个 rebase 只会在同几个文件里反复冲突。冲突：recipe 从已删除的 `config_registry.py` 挪到 `torchtitan_recipes/tests/models/kimi_k3.py`；LoRA 在 MX QAT 之后执行，改成写进 `relations.py`（main 的 #5007）；`LinearCrossEntropyLoss` 用 main 的 loss 构造函数（#5026 之后没有 compile_config）；adapter 测试改用 main 的 `MODEL_FLAVORS` 和 `build_model_config` |
| `2224c7ab0` | #5025（Shuhua，未合） | torch dev20261005 下 FSDP × PP 需要 |
| `84040f93b` `0dc1946aa` `fa07145b8` | #4656 list、remat、PR A | 冲突只在 #4639 的 attention metadata 和 `vision_bank_indices_T` |
| `ca10f9ad0` `7b054f447` `d023f1d93` | #4765、#4764（review 分支 `fb5d35b56` / `908db8609`） | 无冲突 |
| `99edeb9a2` | DEP `6cda7daca` 作为一个改动 | `vision_embeds` 和 `vision_bank_indices_T` 都保留 |
| `cc6885af7` | MoonEP `16ff9da7f` 作为一个改动 | 无冲突 |
| `14c981cff` `888777913` | remat 测试和 DEP 测试适配 main / PR A | 同 `wt_int1006` |
| `1b7e2a6ab` | optimizer 放宽 | 同 `wt_int1006` |
| `6bf639aa6` | Elfie 的 MX QAT、测试、checkpoint 脚本按 main 的接口改 | `MXQATTransform.transform` 接收 context；自冲突声明挪进 `relations.py`；测试用 `ConfigLoader` 和 `build_model_config`。`scripts/validation/kimi_k3/validate_distmuon.py` 还用 #4810 之前的 API，没有移植 |
| `e6baa03e6` | rl flavor（集成树 `7b6959b91`） | 12 层，MLA 在 3 / 7 / 11 层，约 7.55 亿参数 |
| `69c4174bb` | report_arch、k3mini（集成树 `5b001bde1` 的 flavor 部分） | released 布局的 debug checkpoint（`/workspace/k3qat_mm_hf`）在 Elfie 的 adapter 下键全部能映射；A_log 由她的 storage reader 整形，不在 `from_hf` 里做，所以旧的 adapter 测试不再搬 |
| `96805ad44` `6d4ef6791` | #4380 的两个提交 | 无冲突 |

环境：`venv_1006i` = `venv_1006` + Elfie fork 的 torchao（`6352062`，`USE_CPP=0`，`--no-build-isolation`）。

### 和 Elfie 重叠的部分

- QAT：集成树的 `2cf71670e` 做在已删除的 GroupedExperts 上，取 Elfie 的版本。
- to_hf 的 layer-0 占位（集成树 `4fd7388bc`）：Elfie 的 adapter 测试 `test_pipeline_export_synthesizes_placeholders_only_on_layer_zero_stage` 已包含。
- released 格式（集成树 `5b001bde1` 的 adapter 部分）：以 Elfie 的为准。

### 没有搬、需要用户决定的

- **MTP**（集成树 `e68c9af72`）：旧实现用模块级全局变量把 MTP logits 交给 loss，并且拒绝 chunked loss。main 的 DeepSeek V3/V4 已经有一套 MTP 接口（预测元组、`roll_mtp_sequence`、`MTPLoss`、`MTPDecoder`）。K3 接到那套接口上等于重写。
- **LoRA/QLoRA**（集成树约 20 个提交，加 `quantize_lora_dcp.py`，以及 `rl_lora` / `rl_qlora_mxfp4` 两个 recipe）：main 的 LoRA 改成按 handler 组织（#4877、#4995），旧的 QLoRA packed MXFP4 专家做在已删除的 GroupedExperts 上。要搬就得在 GroupedLinear 和 handler 上重新设计；Elfie 的 MXFP4 packed 存储也许能复用。
- **neighbor transport**（集成树 `a7328e4fa` `3128317e1` `3d8e13071`）：main 的 PP 执行层变化很大（#4540 静态 eager 执行、#4662 send 预算），要先确认这个修复在 main 上是否还需要。
  - 10-06 查了：torch dev20261005 的 pipelining 已经自带这两样。`torch.distributed.config.pipeline_per_edge_p2p`（环境变量 `TORCH_DISTRIBUTED_PIPELINE_PER_EDGE_P2P=1`，用 TorchComms 时自动打开）给每条有向的 rank 边一个两 rank 的子 communicator，`(src, dst)` 为键，正反两个方向分开；`_initialize_pipeline_distributed_state` 在第一步之前用一次 parent all-reduce 定 metadata 模式，再预连接 parent 或这些子 communicator（`pipelining/stage.py:193`、`schedules.py:365`、`_p2p.py`）。不用 per-edge 时，torch 会建议把 PP 组建成 `backend="nccl-lazy"`。
  - 差别只有一处：动态 metadata 在 torch 里走这条边自己的 P2P 组（设备上的 `send_object_list`，`stage.py:1969`），不是旧树那样走每个 replica 的 CPU 组；静态 metadata 模式下根本不交换。旧树走 CPU 是为了避开共享 communicator 的 op 顺序，边各有各的 communicator 之后这个理由不在了。
  - 而且 main 自己已经打开了：`init_distributed` 在 PP > 1 时把 `pipeline_per_edge_p2p` 设成 True（`torchtitan/distributed/utils.py:406-417`，#4908 起），torch 没有这个选项就直接报错。5060 上的探针（下面）里，不设环境变量的 PP 格子每个 stage 的 `p2p_per_edge` 也都是 True。
  - 所以旧树的三个提交（eager 建边、两 rank 边组、CPU metadata 和投票）不用搬，recipe 也不用设什么。

## 10-06 白天：5060 上的完整测试

### CPU

- 集成树 `wt_int_elfie`（`6d4ef6791`）全量 `tests/unit_tests/cpu`：1497 passed，17 skipped，1 error。
- main `3f087cf15` 同一个 venv（venv_1006i）：1387 passed，17 skipped，1 error。
- 两边唯一的 error 都是 `test_torch_checkpointing.py` 收集失败（venv 里没有 `torch_checkpointing` 包）。集成树多 110 个测试，没有新的失败。

### GPU 矩阵（kit `kit_int_2026-10-06/`，20 步，每对共用一份 cache，预热也跑 20 步，typecheck 关）

- PP 的对照不能直接用 main：torch dev20261005 下 main 的 PP 在 FSDP `finalize_backward` 报 `requires manual backward finalization`，要 #5025。所以 PP 的对照树是 `wt_main_5025` = main `3f087cf15` + #5025（`eec07d907`，detached，没有建分支）。

| 配置 | 卡数 | 集成树对 main |
|---|---|---|
| dp1 | 1 | 20 步逐位相同 |
| fsdp2 | 2 | 20 步逐位相同 |
| tp2（ep2，SP 开） | 2 | 20 步逐位相同 |
| fsdp2 × ep2 | 2 | 20 步逐位相同 |
| cp2 all-gather | 2 | 第 5 步开始不同：集成树带 #4380，第 5 步第一张 256 patch 的图被切；集成树和 #4380 树（`fcaaeb25f`）默认阈值那格 20 步逐位相同 |
| cp2 Ulysses | 2 | 同上；all-gather 和 Ulysses 20 步互相逐位相同（main 和集成树都是） |
| cp2 all-gather，#4380 阈值 128 | 2 | 只有集成树；和 #4380 树同一格 20 步逐位相同 |

只有集成树才有的功能（20 步，fsdp2，loss / grad norm）：

| 格子 | step 1 | step 5 | step 10 | step 20 | 显存 |
|---|---|---|---|---|---|
| Elfie 的 MX QAT recipe | 8.21236 / 2.3906 | 4.90963 / 6.7500 | 3.50727 / 2.3125 | 3.35226 / 1.9219 | 0.50 GiB |
| rl flavor（7.55 亿参数） | 12.63147 / 15.9375 | 5.51195 / 6.3750 | 3.59533 / 5.5312 | 3.15839 / 2.8281 | 5.15 GiB |
| report_arch flavor | 12.41947 / 5.0625 | 8.69264 / 6.1250 | 6.62681 / 3.7344 | 6.04099 / 3.5781 | 1.11 GiB |
| k3mini flavor | 12.48860 / 10.4375 | 6.23454 / 5.1562 | 4.39495 / 3.4219 | 3.46458 / 2.0625 | 2.63 GiB |

- 四格都跑完 20 步，loss 一直在降。QAT 第 1 步和不开 QAT 的 fsdp2（8.21143）不同，这是 QAT 改了前向，预期如此。
- 集成树第 1 批的 pp2 两格不算：main 那格是上面的 #5025 问题；PP 下 rank 0 不是最后一个 stage，打出的 loss 是 −1，kit 的 `run_matrix2.sh` 改成可以指定 `LOG_RANK`（最后一个 stage 的 rank），PP 格子都重排到第 2 批。

第 2 批（4 卡，20 步，typecheck 关；PP 的对照是 main + #5025，`LOG_RANK` 取最后一个 stage 的 rank）：

| 配置 | 集成树对 main |
|---|---|
| fsdp4 | 20 步逐位相同 |
| cp2 × fsdp2 | 第 1 步就不同：DP 组 1 第 1 步有一张 256 patch 的图，#4380 把它切了；集成树和 #4380 树同一格 20 步逐位相同 |
| fsdp2 × pp2（vpp2） | 20 步逐位相同 |
| fsdp2 × pp2 × ep2 | 20 步逐位相同 |
| pp2（vpp2，2 卡） | 20 步逐位相同 |
| pp4 × vpp4，M8 | 20 步逐位相同 |

pp4 × vpp4（M8）上集成树才有的开关，和 main + #5025 的 pp4 × vpp4 比：

| 开关 | 20 步 | 生效的证据 |
|---|---|---|
| DEP（`vision_dep.enabled`） | 逐位相同 | 日志 "vision encodes decoupled"，"8 micro-batch(es) with images; encodes 8 before the schedule" |
| DEP + bubble | 逐位相同 | "encodes 5 before the schedule, 3 in idle slots; backwards 3 in ..." |
| PR A `cpu_offload="all"` | 逐位相同 | 探针：3 步里 rank 1 / 2 / 3 往 host 存了 120 / 24 / 72 个张量（120 / 24 / 72 MiB） |
| PR A `cpu_offload="planned"` | 逐位相同 | 探针：host 88 / 24 / 24 个；rank 3 的计划 "0 stage micro-batch(es) moved, peak 0.47 -> 0.47 GiB, target 0.54 GiB"（rank 3 本来就在目标以下） |
| balance（mooncake tcp） | 逐位相同 | 探针：rank 1 / 2 往别的 rank 的 pool 存了 48 / 16 个张量（48 / 16 MiB） |
| `TORCH_DISTRIBUTED_PIPELINE_PER_EDGE_P2P=1` | 逐位相同 | 不算对照：main 已经默认打开（见上），探针里两格的 `p2p_per_edge` 都是 True |

- 探针是 kit 的 `local/feature_probe/sitecustomize.py`：包住 `HostBackend.put`、`RemoteBackend.put` 计数，记下每个 stage 的 `p2p_per_edge`，退出时写到每个 rank 的文件。rank 0 的文件是空的，因为 torchrun 的父进程也加载了 sitecustomize，退出时用空计数覆盖了它；rank 1-3 就够了。
- MoonEP 在 5060 上跑不了（没有 switch multicast），它的 GPU 单测会 skip；集成树里的 MoonEP 是 `16ff9da7f`，H100 上 10-06 验过。
