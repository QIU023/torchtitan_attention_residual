# #4380 结构问题和按 DEP 方式的重构方案（2026-10-07）

用户："审核一下昨天重构DEP的方式并且记录到规则，我觉得目前的Dynamic CP head貌似结构不太好"。规则已写进 `CLAUDE.md` 的 "Mechanism structure: the vision_dep pattern"。

## DEP 的重构方式（`k3_pp_mm` = `6cda7daca`），审查结论

做对的地方：
- **一个机制一个包，一个文件一种职责**（`kimi_k3/pipeline_parallel/vision_dep/`），照 4312 的 `cache.py` / `stage.py` / `layout.py` 拆：
  - `plan.py` 是纯规划，不 import `torch.distributed` 和模型；
  - `runtime.py` 管每个 rank 的状态和通信；
  - `stage.py` 是 `PipelineStage` 的薄子类，`schedule.py` 是调度包装；
  - `__init__.py` 负责接线，也是模型唯一 import 的入口。
- **机制类不依赖模型类：** `VisionDepPipelineStage` 直接继承 `PipelineStage`。K3 在自己的 `pipeline_parallel/__init__.py` 里用 `_VisionDepAttnResStage(VisionDepPipelineStage, AttnResPipelineStage)` 把两者组合起来。
- **模型文件只多三样东西：** `KimiK3VisionDepConfig` 配置字段、forward 的 `vision_embeds=` 入口、`pipeline_kimi_k3` 里一次 `install_vision_dep` 调用。
- **长函数拆成命名步骤：** `VisionDepPlan._build` 依次调 `_place_encodes`、`_place_backwards`、`_anchor_work`、`_hook_transfers`。
- **重构不改行为，且有证据：**
  - 搬动的代码只差改名；
  - 规划器在 32,096 组输入上和旧版等价；
  - 玩具流水线 A/B 逐位相同；
  - 重新组合的类做了方法指纹比对（56 个方法解析到同一份代码）。

还没做到的（不急，记下来）：
- `runtime.py` 和 `__init__.py` 用 `MoonViTEncoder` 做类型注解。实际只按 `(pixel_values, grid_thw)` 调用，所以只是注解层面的耦合。
- 测试走的是 K3 组合出来的 `_VisionDepAttnResStage`，没有单独测普通的 `VisionDepPipelineStage`。
- 包还在 K3 目录里：按规则，等第二个模型（K2.5）能跑 DEP 时再 `git mv`。

## #4380 head（`9b03b4af1`）的问题

- **机制全塞进了 `kimi_k3/vision_encoder.py`（+278 行）：** `VisionCPLayout`、`_all_gather`、`_gather_bands`、`KimiK3VisionCPAttention`、`build_cp_subgroups`、`_band`、`_split_mask`，以及 `_forward_split`。
- **纯规划是模型目录下一个零散文件 `kimi_k3/vit_cp_plan.py`**，不在包里。
- **类名带 K3**（`KimiK3VisionCPAttention`），但这个类只依赖 common 的 `VisionAttention`。
- **`_forward_split` 有 95 行**，里面嵌了 `band_rows`、`band_merged` 两个闭包，打包输入、跑 block、拼 bank 三件事写在同一个方法里。
- **`model.py` 的 `parallelize` 里直接写了建子组的逻辑**，而 DEP 只有一行 install 调用。

## 按 DEP 方式的重构方案（纯搬移，加上拆方法，数值不变）

新包 `torchtitan/models/kimi_k3/vision_cp/`：

| 新文件 | 内容 | 来自 |
|---|---|---|
| `plan.py` | `ImageShard`、`row_partition`、`subgroup_layout`、`balance_images`、`classify`、`DynamicCPPlan`、`plan_dynamic_cp`、`key_runs`，原样 | `vit_cp_plan.py` |
| `attention.py` | `VisionCPLayout`、`_all_gather`、`_gather_bands`、`VisionCPAttention(VisionAttention)`（改名，去掉 K3） | `vision_encoder.py` |
| `encoder.py` | `VisionCPEncoder` mixin：`forward` 分派，加上 `_forward_split` 拆成的 `_pack`（整图和行带、位置表、RoPE、query 的 runs）、`_encode`（block 和 projector）、`_assemble_bank`（补齐、CP all-gather、拼回），以及 `_band`、`_split_mask`；只用 `MoonViTEncoder` 的公开部分和 `_tpool_patch_merger` | `vision_encoder.py` |
| `__init__.py` | `build_cp_subgroups`，以及 `install_vision_cp(model, parallelism_context)`（建子组，交给 tower） | `vision_encoder.py`、`model.py` |

K3 目录留下的：
- `vision_encoder.py`：`KimiK3VisionEncoder(VisionCPEncoder, MoonViTEncoder)` 和 `dynamic_cp_min_patches` 配置字段，其余回到 main 的样子；
- `flavors.py`：换用 `VisionCPAttention`；
- `model.py` 的 `parallelize`：`if cp_enabled: install_vision_cp(self, parallelism_context)` 一行。

测试：
- `test_kimi_k3_vit_cp_plan.py` 改名为 `test_kimi_k3_vision_cp_plan.py`，只改 import；
- GPU 测试只改 import；顺带把开头两行的注释缩成一行。

数值不变的证据（照 DEP）：
- 搬动部分的 diff 只差改名；
- `_forward_split` 拆成的三步和原函数体逐行对应；
- encoder 和 attention 的方法指纹和旧 head 比对；
- GPU 会话：2 个 GPU 测试，加上 CP=2、阈值 128 的一个端到端格子在同一份 cache 上 20 步轨迹和 `9b03b4af1` 逐位相同。

## 已做（10-07，用户："直接改，review和pr分支都做"）

- `k3_cp_mm` = `cpmm_review1` = `6317c5538`：
  - `e094d6b9a` 按上面的方案重构；
  - `6317c5538` 把测试注释缩成一行。
- 和方案的差别：`VisionCPEncoder` mixin 改成单继承的 `MoonViTCPEncoder(MoonViTEncoder)`。这样阈值配置跟着机制走（`MoonViTCPEncoder.Config.dynamic_cp_min_patches`），也避免 mixin 访问父类属性带来的类型检查问题。
- 证据见 `PR_BODY_CP_MM_v3_2026-10-06.md` 的状态区：
  - 9 个搬过去的函数 AST 一致；
  - gloo 上新旧两版 368 个张量逐位相同；
  - CPU 7 passed。
  - 复现脚本在 `kit_cpmm_2026-10-06/local/vision_cp_equiv/`：`kda_stub.py` 只给 Windows 用；跑法是 `python dcp_equiv.py <树> <输出> old|new`，再 `python dcp_compare.py <输出>`。
