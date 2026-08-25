# 交接:k3_on_4025 清理完成状态,与切分前剩余工作(2026-08-25)

Windows 侧完成的清理都在 `k3_on_4025`,已推。GPU 侧接手:剩余合并/排序 → 按行切分
到轴分支 → 重跑 PR evidence。

## 已落在 k3_on_4025 的 commit(全部 message 无 trailer、无 #N,committer=QIU023)

| commit | 内容 |
|---|---|
| `b32b0fa1` | model 侧 `_debugmodel` 参数化;`_debugmodel_text_32l` 塌到 6 行(删两行死代码) |
| `4e87481b` | `pipeline_kimi_k3_with_cache_adapter` -> `pipeline_kimi_k3`(4 处);trainer 侧 `kimi_k3_debugmodel_text`/`_text_32l` 派生化(base + 换 model_spec/dataloader + **重算 loss vocab**) |
| `ef206e0e` | `pp_cleanup.py` 脚本应用:`_DBG`/`_dbg` 通道删除、非 ASCII 清零、粗体 markdown 清除、多余空行 |
| `825cf705` | `adapter_enabled()` 的 30 行实验叙事 -> 3 行门槛陈述 |
| `55a773bc` | 全文件夹扫除(本文档主体,见下) |

## `55a773bc` 做了什么(净 -215 行)

1. **仓库外引用清零**:`phase13_*` / `Raising_PRs/*` / logbook / matrix_scripts 的文件名
   引用全部删除(pipeline_adapter 3 处、kcp / dep_bubble_runtime / vit_prefetch /
   sharding 各 1 处、dep_bubble_plan 1 处、knobs 1 处)。"spec section 4.1" 一类内部
   规格编号改为自包含表述。验证:`grep -rn 'phase13|Raising_PRs|logbook|matrix_scripts'
   torchtitan/models/kimi_k3/*.py` 为空。
2. **env 变量退役**:knobs.py 重写为 config-only(`resolve_knob`/`_field` 的 env 分支、
   `import os`、弃用警告全删,~198 -> ~115 行);所有 user-facing 报错里的
   `KIMI_VIT_*` / `TORCHTITAN_ATTNRES_CACHE` 改为 config 字段名。
   **行为变化仅限"env 覆盖 config"这条路径**;config 驱动的运行不变。
   `vit_bubble_max_pending` 缺省 0 逐字保留(第一版草稿险些写成 2,靠 Write 的
   读前检查挡住 -- 重写文件前必须读当前分支版本)。
3. **import 上提**:pipeline_adapter 全部函数级同仓库 import 移到文件顶
   (knobs/layout/dep_*/vit_prefetch/distributed.pipeline_parallel/inspect)。
   环已排除:这些模块没有一个反向 import pipeline_adapter(layout 的引用只是
   docstring 文字)。`get_schedule_class` 从 titan 侧以 `_tt_get_schedule_class`
   别名引入,避免与顶部 try 块里 torch 版本探测的同名冲突。
4. **叙事 -> bullets**:module docstring、grad-bridge 注释块、shape-inference、
   mark_no_hook、capture_grad、keepalive、on_microbatch_end、patched_bwd、
   _unwrap_multimodal、_register_mm_prefix_hooks、dep_vision_stages。原则:
   保留不变量与约束,删调试过程("之前两版怎么错的"一类)。
5. 死常量 `_ATTN_RES_EXTRA_LAST_STAGE_FQNS` 与三连分节横幅删除。

## 用户已定的决定(不要重新讨论)

* **不建 parallelize/ 子目录** -- 上游六个模型零先例,规则文件写死平铺结构。
* **合并方案**(用户批准,未执行,是 GPU 侧下一步):
  `dep_bubble_{plan,runtime,backward}.py + dep_vision_stage.py + vit_prefetch.py`
  -> 一个 `dep.py`(~1226 行);`layout.py` -> 并入 `pipeline_adapter.py`。
  `kcp.py`/`vit_cp_plan.py`/`dtensor_ops.py` 保持平铺。
* **knobs.py 保持独立文件**:消费者是 pipeline_adapter 和 vit_prefetch(合并后是
  dep.py)两方;并入 adapter 会造出 dep->adapter->dep 的顶层 import 环。
* **文件内排序**:纯文本的函数/类放上面,vit/多模态放下面 -- 为最终按行切分服务。
  pipeline_adapter 现状已大致如此(adapter 本体在前,FQN+DEP 接线在后)。
* **import 规矩**:同仓库可 import 的一律文件顶;函数内 import 只允许不同 binary
  依赖(如 fla)。
* PP body 叙述以 adapter 设计为中心;CP/EP 是标准审阅件,不需要设计叙事。
* PR 分支(k3_pp_text / k3_cp_text / k3_ep)**只有用户可推**;k3_on_4025 可推。

## GPU 侧待办(顺序)

1. **合并**(上表方案),同时做文件内 text-top 排序。纯移动,判据 = 58 格逐位。
2. **按行切分**到轴分支:文本函数在上、vit 在下的排布就是为这一步准备的。
   切分后每个轴分支重跑各自 evidence(PP 9 格、CP cp2/4/8、EP ep2/4/8),
   body 的数字必须来自切分后的 head。
3. **CP/EP 分支的同类清理**:用户只看完了 PP;CP/EP 的注释审查在用户处进行中,
   等指示。两分支同样带 `Claude-Session` trailer 不一致问题(PP 主 commit 无、
   其余有),force push 重切时统一(先例:v2 重切时上游向 commit 不带 trailer)。
4. tests/integration_tests/features.py 里我们的 cell 注释仍有一个非 ASCII 箭头
   (L132 区域),顺手清。
5. `sharding.py` 顶部 docstring 剩余部分未审(只删了 logbook 引用行);
   `vit_cp_plan.py`/`dtensor_ops.py`/`layout.py` 未做叙事审查 -- 合并时一起。

## 验证基线

* 每步 `py_compile` 通过;残留扫描
  (`phase13|Raising_PRs|logbook|matrix_scripts|KIMI_VIT|TORCHTITAN_ATTNRES|_dbg|_DBG|resolve_knob`)
  全零;非 ASCII 全零(features.py 那一处除外,见待办 4)。
* 本轮全部是注释/命名/构造重构 + env 路径删除:**config 驱动的运行应逐位不变**,
  58 格 gate 是判据。knobs 的 env 路径删除意味着任何还在 export 旧变量的脚本
  将静默失去效果 -- gate 脚本已全部走 config(2026-08-2x 的 finding-32 迁移),
  但 GPU 侧启动器里若有残余 export,现在是清掉的时机。

## Windows 侧文件位置

* 清理脚本:`phase13_k3like_48b_posttrain/cleanup_scripts/pp_cleanup.py`
  (GPU 写的;Windows 上要 `PYTHONUTF8=1` 跑,pathlib 缺省 GBK 会炸)。
* 我的分步脚本在 Windows 临时目录,内容已全部落进 commit,无需保留。
