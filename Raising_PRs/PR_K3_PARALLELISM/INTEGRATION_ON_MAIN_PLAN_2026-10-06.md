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
