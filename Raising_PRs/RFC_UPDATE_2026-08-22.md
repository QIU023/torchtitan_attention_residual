# RFC 主体更新(2026-08-22)

接 `RFC3029_UPDATE_2026-08-04.md`(主体)与 `RFC_PASTE_FAITHFUL_2026-08-05.md`
(矩阵小节的替换块)。这份是自 08-04 以来的第一次主体更新。

三件事变了:上游 main 合并进了 fork(4025 本身未合并,PASTE 里已写明)、TP 从命令式计划迁到声明式、上游开了两个 K3 tracking
issue(4272 总表,4269 是 MXFP4/MXFP8 QAT 与 checkpoint 互操作,当前无人认领)。

格式按 `CLAUDE.md` 的 PR-text 规则:单行段落(表格与列表除外)以便逐字复制,
正文无小标题、无粗体结构、无证据表格。证据留在本文档的"支撑材料"一节,
以及 `ISSUE_4269_GAP_AUDIT_2026-08-22.md`。

--- PASTE BEGIN ---

Rebased onto current torchtitan main and re-validated: 58 parallelism configurations -- FSDP2/HSDP, TP, PP (incl. interleaved 8x4), CP, EP and their combinations, each across a text, a multimodal, and a multimodal+LoRA debug model -- train 10 steps under `--debug.seed 42 --debug.deterministic`, all passing. The upstream K3 reference-model PR has not merged; when it does, the model half of this rebases onto it.

Complete and review-ready:
- the model -- KDA + MLA + latent MoE + Block Attention Residuals + MTP -- loading the released 48B weights
- TP declared on the module configs following deepseek_v3; the migration is behaviour-free -- every non-LoRA configuration bit-identical, including all fourteen with TP
- PP carrying the block residual across stages through an adapter private to the model's parallelize, validated to 8 stages x 4 virtual; CP as Ulysses on the MLA layers and a rank-to-rank state pass on the KDA layers; EP; four-axis combinations of the above
- LoRA on all of the above, adapter sharding derived from the base layer's declaration
- MXFP4-weight / MXFP8-activation QAT semantics (fake-quant, straight-through backward) under FSDP and TP, and streaming quantize-then-shard import of the packed checkpoint -- no rank materialises the full bf16 model
- quantile expert balancing: distributed aggregation, offline convergence (expert-load cv 0.607 -> 0.053), and the TP interaction fixed

Working and validated, pending migration to the idiomatic form:
- the TP remainder: the per-layer AttnRes norms, packed-MXFP4 under TP, and the module-boundary unwrapping (`use_local_output` / `to_local`) -- removing it is the `spmd_types` conversion
- the vision tower: its TP and dynamic CP are imperative and model-private

Not done, and what blocks each:
- all five axes in one run: 32 ranks at degree 2, we have 8
- MoonEP token dispatcher: interface draft, dispatch and combine raise; needs NVLink hardware (a 2-GPU NVLink pair covers the correctness half)
- packed checkpoint export (DCP -> HF packed safetensors), and a byte-level codec comparison against compressed-tensors
- the 2.8T configuration: config-level only, never run -- validation is on 48B real weights and a K3-faithful topology

--- PASTE END ---

## 支撑材料(不进 RFC 正文)

| 声称 | 证据 |
|---|---|
| 58/58,非 LoRA 逐位不变 | `phase13_k3like_48b_posttrain/gate_logs/gate_58_2026-08-22_behaviour_free_percell.txt` |
| 判据本身与一次被撤回的放宽 | `phase13_k3like_48b_posttrain/TP_DECLARATIVE_2026-08-21.md` |
| 词表声明缺失 | `phase13_k3like_48b_posttrain/TP_GRADNORM_INFLATION_2026-08-21.md` |
| CP 契约(Ulysses / KCP) | `phase13_k3like_48b_posttrain/CP_DECLARATIVE.md` |
| 4269 逐条差距 | `Raising_PRs/ISSUE_4269_GAP_AUDIT_2026-08-22.md` |
| quantile 收敛 | `phase13_k3like_48b_posttrain/QUANTILE_BALANCING.md` |

## 写这份时刻意收掉的三个说法

**"5D"**。上游 4272 写的是 "FSDP/HSDP, TP, EP, CP, and PP together",五个。
我们最高是四轴同开;五轴一次都没跑过,8 卡上 2^5=32 > 8,物理上跑不了。
说四轴是事实,说 5D 会被一个问题问穿。

**"2.8T 可以直接 smoke"**。按 `CLAUDE.md` 的诚实规则,只能说
"validated on 48B real weights and a K3-faithful topology; 2.8T 是 config-level,
从未运行"。给了卡之后能不能 smoke 是预期,不是已验证的能力。

**"QAT 覆盖了 MXFP4/MXFP8"**。是 fake-quant + STE,bf16 计算。
写成"MXFP4 权重与 MXFP8 激活的 QAT 语义"而不是"MXFP4/MXFP8 训练"。

## 与上一版(08-04)相比新增的可声称项

* 上游合并完成,矩阵在合并后的树上重跑
* TP 声明式迁移,且是可证明的 behaviour-free 重构
* 三个缺陷的定位与修复(其中两个 58 格永远看不到)
* QAT / packed 导入 / quantile 的首次矩阵级验证(以及由此查出的两个失败)

## 与上一版相比需要收回或弱化的

* 08-04 那版把矩阵结果按"twin 对照"呈现;`RFC_PASTE_FAITHFUL_2026-08-05.md` 已经
  因为上游模型结构变化而撤掉了 twin 那一列。本次不恢复。
* `ALIGN_TO_4025_2026-08-13.md:22` 写的 "Everything TP is declared on the Config
  tree" 对 4025 的 K3 树不成立 —— 他们那棵树上 sharding 声明是 0,
  `parallelize.py` 只有 100 行且 TP/PP/CP/EP 全部 `NotImplementedError`。
  那句话描述的是 core 里 llama3/deepseek 的机制,不是 4025。RFC 正文不要引用它。
