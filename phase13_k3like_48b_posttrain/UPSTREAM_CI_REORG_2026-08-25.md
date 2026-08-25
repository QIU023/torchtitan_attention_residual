# 上游 CI 重组(2026-08-25),以及它让我们的测试静默失效

上游 `b9e447a61 "[ci] Run fake-PG for functionality tests, add numerics guard for
models, enable GPU unittests"`(wwwjn, 2026-08-24)重做了测试分层。两条后果都是
**静默**的 —— 不报错,只是不跑。

## 一、`tests/unit_tests/` 顶层不再被收集

`unit_test_cpu.yaml` 现在是:

    pytest tests/unit_tests/cpu --cov=. ...

`unit_test_gpu.yaml` 是 `pytest tests/unit_tests/gpu`。**顶层的文件两个 workflow
都不收**。而 `upstream/main` 顶层还留着:

    test_flex_attention.py  test_kimi_k3.py  test_model_td_layout.py
    test_packed_vision.py   test_qk_clip.py  test_varlen_attention.py
    test_vision_attention.py

—— 这些是 4025 在 split 之后合进来的,现在上游自己也跑不到。我们三个新单测原本
放在同一层,同样跑不到。已全部移进 `cpu/`:

| 分支 | 文件 | 结果 |
|---|---|---|
| `k3_pp_text` | `cpu/test_kimi_k3_pp_fqn_injection.py` | 4 passed |
| `k3_cp_text` | `cpu/test_kimi_k3_cp_contracts.py` | 5 passed |
| `k3_ep` | `cpu/test_kimi_k3_ep_sharding.py` | 3 passed |

## 二、integration test 分 fake-PG / real-PG 两层

`OverrideDefinitions` 新增 `use_real_pg: bool = False`,`validate_fake_pg_compatibility`
在构建列表时检查:

* `checkpoint.enable` 或 `create_seed_checkpoint` → 需要 real PG
* `pipeline_parallel_degree > 1` → 需要 real PG

不满足就 **`raise ValueError`**,不是 skip —— 整个 features suite 构建失败。所以
PP 的两格必须标 `use_real_pg=True`。

CP / EP 不在这个清单里,fake PG 下能"通过",但 fake PG 的 collective 是 no-op:
cp2 格不会真的走 all-to-all 和 KCP 状态传递,ep2 格不会真的走 token dispatch。
格子绿着,却什么都没验证。两格也标了 `use_real_pg=True`,进 8 卡 real-PG 层。

三棵树各自跑了一遍全 features 列表的 `validate_fake_pg_compatibility`
(47 / 46 / 46 个 config),无一 raise。

## 三、rebase 本身

`upstream/main` = `30eb5e502`,比我们分支多 6 个提交,三个分支 rebase 零冲突。

另一个必须先 rebase 再跑证据的理由:`6f791d775 "[data] TextCollator: pad positions
within the context window"` 动的是数据侧,可能移动 loss。

**实测没有移动**:`kimi_k3_debugmodel_text_32l` 的 dp1 在 rebase 前后

    s1=12.47883  s3=7.56594  s10=3.59412

三个数完全一致。但这是事后才知道的 —— 判据是"证据必须属于将要提交的那份 diff",
不是"我猜它不会变"。

## 四、还没做的

`k3_on_4025`(集成树)、`k3_pp`、`k3_cp`(两个多模态分支)还在旧 base 上,单测也
还在顶层。见 task #27。
