# debug model 放宽度的定尺寸（2026-09-28，5060）

用户："理想情况，在能扩宽度的情况下，从debugmodel开始扩比较合适 ... 把PR A和DEP定下来吧"。给 `H100_SIZES_2026-09-27.md` 第 1 到 3 节定宽度；结果只用来定尺寸，不进 body。

- 模型：main `f35966713` 的 `_debugmodel`，和 dim 成比例的宽度都乘 s = dim / 256（头数、q_lora、kv_lora、dense hidden、latent、expert hidden、视觉塔的 dim / qkv / hidden / 头数），17 层、block 4、8 专家 top-2、vocab 2048 不变。
- `widen_debug.py`：meta 上数参数，打印每个 rank 的静态（按 16 B/参数；实测对得上的是 8 B/参数，因为 recipe 的 `training.dtype` 是 bf16），输出 `widen_debug_pp4vp2.out.txt`。
- `probe_lbw.py`：`kit_pp_lowerbound_2026-09-26/probe_lb.py` 换成放宽度的 debug model（PR A 的布局：AdamW、FullAC、M16、seq 2048）。
- `dep_widened.py`：已提交的 `kimi_k3_debugmodel_pp4_vp2_vit_dep` 只换模型（本地替换 `config_registry.model_registry`），记录每个 rank 每步的峰值。
- `sizing2.sh`：pp4 × vp2，dim 512 / 768 / 1024 各 4 步；PR A 布局在 main 上，DEP 格在 DEP 的 head `31f372593` 上；本地 torch 兼容补丁跑前打上、跑完撤掉（`results/worktrees_after.txt`）。
- `fit3.py`：每个 rank 按 A + Bs + Cs² 过三个点，外推到 H100 的宽度，输出 `fit3.out.txt`（allocated 和 reserved；reserved 在小尺寸上受分配器粒度影响，外推偏高，只作参考）。
- `ratio.out.txt`：放宽度后视觉塔和文本每层激活参数的比例（不随宽度变）。
