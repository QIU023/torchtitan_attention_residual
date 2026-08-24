# 四个并行 PR 的当前状态(2026-08-24)

树:`k3_on_4025` @ `0a578a0ad`,基于上游 4025 head `ded1bf109`,28 个提交,已推到
fork `QIU023/torchtitan`。

## 一、上游测试:零回归(实测)

| | |
|---|---|
| 我们分支 `tests/unit_tests/` | 28 failed, **729 passed**, 22 skipped |
| 上游 head `ded1bf109` 同样 7 个文件 | **28 failed**, 48 passed |
| 失败用例 ID 差集(我们 - 上游) | **空** |

失败集中在 `test_helion_rope`(15)、`test_distributed_linear`(4)、
`test_model_td_layout`(3)、`test_varlen_attention`(3)、`test_dist_gemm`、
`test_qwen3_5_deltanet`、`test_rope` 各 1 —— 与 kimi_k3 和 vision_encoder 无关,
且在上游 head 上逐条同样失败。另有 3 个文件 collection error
(`test_download_hf_assets` / `test_tokenizer` / `test_torch_checkpointing`,
缺 `torch_checkpointing` 等模块),同样在上游 head 上复现。

对照方式:`git worktree add --detach`,**不用 stash**(用 stash 切分支会捞到别的
分支的旧 stash,已踩两次)。

## 二、各轴状态

| 轴 | 实现 | 门(33 格全暖) | 判据口径 |
|---|---|---|---|
| **CP** | 文本侧 MLA Ulysses + KDA KCP;视觉 dynamic CP 已按老树重搬(子 CP 组 + 负载均衡) | `cp2` 三臂全过 | 未达标:2 步,无归档基线 |
| **PP** | 块残差作为 stage 载荷 | `pp2/pp4/pp8` 三臂**全部与 dp1 逐位相同** | 同上 |
| **DEP** | 塔独占 stage(clause 1)+ 塔跨 stage 分解(clause 2) | 端到端未跑 | share 分解已逐位等价(`atol=0`) |
| **EP** | AllToAll dispatcher(不含 MoonEP) | `ep2_fsdp2` / `ep8_fsdp8` 全过 | 未达标 |
| **TP** | 暂停(maintainer 要求) | — | — |

**"32/33 通过"不是数值判据。** 老树 `TP_DECLARATIVE_2026-08-21.md` 明确写过
「"58/58" 不是数值判据,它只表示每格跑满 10 步」,且"有差异的格子必须有根因"
这条放宽**已被撤回**。当前这轮是 **2 步**(老树 `run13_flav.sh` 是 10 步),
且新树没有归档基线可做同格前后比对,所以**按老树判据不合格**。
`lora_pp8` 还挂着(pp8 下某 stage 无 LoRA 可训练参数),判据第一条"0 格挂掉"
本身就不满足。

## 三、分支切分:naive cherry-pick 不成立(实测)

`BRANCH_CUTS_2026-08-23.md` 写「唯一的真冲突是 `unsupported_parallelisms` 那个
list,三个分支各删一行,一分钟的手解」—— **低估了**。

实测:从 `ded1bf109` 起 cherry-pick EP 的第一个提交 `418d22448` 就冲突:

    CONFLICT (content): Merge conflict in torchtitan/models/kimi_k3/parallelize.py

因为 EP 的提交是写在 CP 的提交之上的。按文件切同样不行 —— 四个轴共同修改:

| 文件 | CP | PP/DEP | EP |
|---|---|---|---|
| `model.py` | ✓ | ✓ | ✓ |
| `parallelize.py` | ✓ | ✓ | ✓ |
| `__init__.py` | ✓ | ✓ | ✓ |
| `config_registry.py` | ✓ | ✓ | |

各轴独占的文件(可干净归属):

* CP:`kcp.py`、`sharding.py`、`dtensor_ops.py`、`kda.py`、`vit_cp_plan.py`、
  `vision_encoder.py`、`common/decoder.py`、`common/vision_encoder.py`
* PP/DEP:`pipeline_adapter.py`、`dep_bubble_{plan,backward,runtime}.py`、
  `dep_vision_stage.py`、`attn_res.py`、`layout.py`、`knobs.py`
* EP:`moe.py`

**可行的切法**(未执行,待定):每个分支从 `ded1bf109` 起,取该轴的提交并**手工
解决 `model.py` / `parallelize.py` 的冲突**,而不是指望 cherry-pick 干净落地。
CP 10 个提交、PP/DEP 6 个、EP 4 个。

## 四、待办(按优先级)

1. `n_vit=2` 端到端多卡验收 —— 复现老树的 n_vit=1 vs n_vit=2 等价性
2. 清 `pipeline_adapter._parallelize_with_tower` 里的 wrapper 残留
   (`KimiK3MultimodalModel.from_parts`,新树无此类)
3. **10 步基线重跑并归档**,此后按同格前后逐位比对
4. `lora_pp8`
5. 分支切分(见上,需手工解冲突)
