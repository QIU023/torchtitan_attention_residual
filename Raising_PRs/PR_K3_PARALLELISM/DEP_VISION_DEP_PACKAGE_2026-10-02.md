# DEP 拆成 `vision_dep/` 包（2026-10-02，CPU 这边）

用户 10-02：
- "目前dep PR的代码有点难以理解了，dep plan文件，有可能按照tianyu在4312 comment针对cache/hook方案重构为attnrespipelinestage的方式重构吗？参照cache.py stage.py layout.py等 但是不能影响数值"
- "粗这个没办法，我们的测试场景就不标准, 不管了"
- "文件名字改一下，加个vision_dep 文件夹，里面是 plan.py __init__.py stage.py  然后把类拆开 和pp cache 4312拆法一样"

## 结果

- review 分支 `dep_review1`：`a93cd48ea` → `b2a57dff7`，快进，一个提交（作者 QIU023，无 trailer）。
- PR 分支 `k3_pp_mm` 没动，仍是 rebase 前的 `d27839459`。等你本地验证后再同步，同步时连同 rebase 到 main `db050eb3f` 的那 5 个提交一起。
- 这个提交只改结构，不改行为。规划器的算法没动（"装不下就退回前后"仍在，按 `DEP_FIX_PLAN_REVIEW_2026-10-01.md` 的结论先不改）。

## 新结构，对照 4312

| 新文件 | 内容 | 4312 里对应的 |
|---|---|---|
| `vision_dep/plan.py` | `VisionDepPlan`：`__init__` 读调度（stage 0 消费和就绪的槽、各 rank 的空闲段、每个 micro-batch 的代价），`_build()` 依次调 `_place_encodes`（前 PP 个进 prologue，其余进空闲段，放不下的退回 prologue）、`_place_backwards`（放不下的进 epilogue）、`_anchor_work`（按 rank 和开始时间挂到锚点动作上）、`_hook_transfers`（每次传输两端的挂出点）。`_Run`、`_slot_times`、`_idle_runs`、`_stage0_slots`、`_hook_at`、`_encode_spot`、`_backward_spot` 原样保留 | `layout.py` 的 `BlockLayoutTables`（`__init__` 加 `_build()`） |
| `vision_dep/runtime.py` | `VisionDep`：每个 rank 一份，本 rank 的各 stage 共用（副本、权重同步、编码和反向、传输、梯度归约），加 `_tp_dim` | `cache.py` 的 `PPRankLocalCache` |
| `vision_dep/stage.py` | `VisionDepPipelineStage(AttnResPipelineStage)` | `stage.py` 的 `AttnResPipelineStage` |
| `vision_dep/schedule.py` | `VisionDepSchedule`：step 前后跑视觉阶段 | 4312 没有；一个类一个文件 |
| `vision_dep/__init__.py` | 接线：`build_vision_replica`（原 `pipeline_parallel/__init__.py` 的 `_vision_replica`）、`pipeline_groups`（原 `_pp_groups`）、`install_vision_dep`、`_connect` | `pipeline_parallel/__init__.py` |

- `pipeline_parallel/__init__.py` 里 DEP 的改动（对 main）从 +95 行降到 +50 行左右，只剩 import、`stage_class` 参数和 `pipeline_kimi_k3` 里的调用。
- 测试：`test_kimi_k3_dep_plan.py` 改名为 `test_kimi_k3_vision_dep_plan.py`，`TestDepPlan` 改为 `TestVisionDepPlan`；"计划只由输入决定"那条原来靠 dataclass 的 `==`，改成逐字段比较。`test_kimi_k3_vision_dep.py` 只改了 import 和 `build_vision_replica` 这个名字。

## 数值不变的证据

1. 代码逐字搬移：`runtime.py`、`stage.py`、`schedule.py`、`install_vision_dep` 和 `_connect`、复制塔和分组两个函数，与 `a93cd48ea` 逐行 diff，只有改名（`plan_dep` 和 `DepPlan` → `VisionDepPlan`，`_vision_replica` → `build_vision_replica`，`_pp_groups` → `pipeline_groups`）和两行新 docstring。
2. 规划器等价（`kit_dep_refactor_2026-10-02/equiv_plan.py`）：旧 `plan_dep` 对新 `VisionDepPlan`，32,096 组输入。
   - Interleaved1F1B：pp 1/2/3/4/8 × vp 1/2/4 × 多种 M；均匀和不均匀负载；代价比例 0.001 到 100，含 K3 区间 0.02 到 0.32 和 H100 那轮的 3.61；可训练和冻结；bubble 开和关。
   - 20,000 个随机动作序列：含 I / W 动作、随机空格、随机 stage 0 所在 rank。
   - 7 个输出字段连插入顺序都相同（放进空闲段的工作共 59,626 个）；8,169 组非法输入两边抛同样的异常和消息。0 处不同。
3. 逐位 A/B（`ab_pipeline.py` + `ab_compare.py`）：4 个 gloo rank，pp4 × vpp2 Interleaved1F1B，CPU 测试的玩具模型，6 组配置各 2 步：K2.5 形式、bubble、bubble 加晚到的梯度（代价 0.25）、bubble 加冻结塔、bubble 加 GELU 塔（lr 1e-3）、K2.5 加 GELU 塔。每步的 loss、eval loss、每个参数的梯度和 plan 全字段，旧树（`git archive a93cd48ea`）和新树逐位相同，共 286 个张量，0 处不同。
4. CPU 测试：`test_kimi_k3_vision_dep_plan.py`、`test_kimi_k3_vision_dep.py`、`test_kimi_k3_pp_stage.py`、`test_kimi_k3_pp_block_grads.py` 共 27 passed、28 subtests passed，和重构前的基线相同；两个 DEP 文件单独是 18 passed，就是 body 里的数。
5. pre-commit 的 hook 逐个单独跑（trailing-whitespace、end-of-file-fixer、check-ast、insert-license、ufmt、flake8、pydoclint、codespell），全过；ufmt 只重排了 `plan.py` 的换行。pyrefly 0.45.1 对改动的 6 个源文件只读检查，0 errors（1 suppressed，是原有的 `read-only`）。没跑 pre-commit 的 pyrefly hook，它会全仓删 suppression。

## diff 审计

- 新增注释：没有。diff 里所有 `#` 行都是原样搬移的：license 头、三个常量的说明、`_encode_spot` 和 `_backward_spot` 各一行、`_connect` 一行、`_send_then_run` 一行、副本随机数那一行。
- 新增 docstring：`build_vision_replica`、`pipeline_groups` 各一行（由私有改成公开）；`runtime.py`、`stage.py`、`schedule.py` 的模块 docstring 各一行；`VisionDepPlan` 的类 docstring 由原来 `DepPlan` 和 `plan_dep` 的两段合并。
- 新私有 def：`VisionDepPlan` 的 `_build`、`_place`、`_place_encodes`、`_place_backwards`、`_anchor_work`、`_hook_transfers`。都是原 `plan_dep` 函数体按步骤拆开，不是新逻辑。在 `torchtitan/` 全树 grep 了 `pipeline_order|idle|bubble`：core 的 `distributed/pipeline_parallel.py` 只在一条 warning 里提到 bubble；`experiments/graph_trainer/graph_pp` 读 `pipeline_order_with_comms` 是为了建图和执行动作，不往空闲段里放额外工作。没有可复用的现成实现，4312 的 `layout.py` 就是同样的写法。`_place` 把编码和反向放进空闲段之后的同一组记账合成一处。
- 两个由私有改公开的名字：`build_vision_replica`、`pipeline_groups`。搬进包以后要被 `pipeline_kimi_k3` 跨模块调用，跨模块 import 私有名字会被 reviewer 问，所以改成公开。

## 没做的

- GPU 测试（`tests/unit_tests/gpu/test_kimi_k3_vision_dep.py`，4 卡 NCCL）这台机器跑不了。它复用 CPU 测试的 `_VisionDepChecks`，本身没改；下次在 H100 上跑一遍。
- GPU 那边 H100 的 K3 区间测量（`DEP_K3RANGE_H100_2026-10-02.md`）跑在 `a93cd48ea` 上，因为数值逐位不变，结果对 `b2a57dff7` 同样成立。

## 复现

`kit_dep_refactor_2026-10-02/`：
- `harness/sitecustomize.py`：只给本机（Windows）用，给 kimi_k3 包打桩（缺 CuTeDSL），并补上 nightly 的 `step(arg_mbs=...)`；不进任何提交。
- 规划器等价：先 `git show a93cd48ea:torchtitan/models/kimi_k3/pipeline_parallel/dep_plan.py > old_dep_plan.py`（放在脚本旁边），再 `python equiv_plan.py <新树>`。
- A/B：`git archive a93cd48ea torchtitan tests | tar -x -C old_tree`；两棵树分别跑 `DEP_ROOT=<树> DEP_STUB_PP=0 PYTHONPATH="harness;<树>;<树>/tests/unit_tests/cpu" python ab_pipeline.py <树> <输出目录>`，再 `python ab_compare.py <旧输出> <新输出>`。
