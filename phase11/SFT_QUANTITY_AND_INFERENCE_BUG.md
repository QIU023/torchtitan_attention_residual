# SFT 量 + 推理 NaN bug 调研

## 1. SFT 量标准（更新前的 16% 估值是错的）

| 算法 | 数值 |
| --- | --- |
| 按 token: 39M / 266M (LLaVA-1.5 mix665K) | 14.7% |
| **按 epoch: 1200 步 / 2344 步 (1 epoch on 150K)** | **51.2%** |

**经验法则**：
* 小 LM (<1B): SFT 1-3 epoch
* LLaVA-1.5 paper recipe: **1 epoch** on instruct mix
* 我们差 1144 步到 1 epoch，约 ~3.5h GPU 时间

## 2. 推理 NaN bug — 真正根因

| 测试 | 结果 |
| --- | --- |
| LM-only greedy "Once upon a time, there was a" | ✅ 真实英文 |
| LM-only greedy "Hello, my name is" | ✅ 真实英文（之前那次） |
| LM-only greedy 重跑 "Hello, my name is" | ❌ `!!!!!!!!!!` (非确定！) |
| VLM greedy 任何 prompt | ❌ `!!!!!!!!!!` |
| 显式查 logprob | NaN logprob, token id 0 |

**真因**：在 Llama-3.1 tokenizer 中 **token id 0 = `!`**（不是 BOS！— 用 `tok.decode([0])` 验证）。

模型 forward 在某些路径下产生 NaN logits → `argmax(NaN_array) → first index = 0 → !`。所以 `!!!!` 是 NaN-fallback 的可见征兆，不是模型"偏好"。

**为什么 VLM 永远 NaN，LM-only 偶发**：
* LM-only：KDA+MLA+MoE 在某些 prompt 长度/内容下有数值不稳定
* VLM：vision feature splice 把数值不稳定区域扩大（每个样本 196 个 image token 把 KDA 的 recurrent state 推向溢出）
* 模型欠训（VLM SFT 51% of 1 epoch）让 unembedding 的 fp16/bf16 logits 在某些区域更接近溢出边界

## 3. 优先级修复路径

| 路径 | 代价 | 收益 |
| --- | --- | --- |
| **A. 续训 SFT 到 1 epoch (1100 步, ~3.5h)** | GPU 3.5h | 大概率消除 VLM 全 `!` 问题（更稳定的 unembedding 分布） |
| B. 数值稳定性修补 (force fp32 in unembed, RMSNorm clamp) | 半天调试 | 治标，仍受欠训影响 |
| C. A + B 都做 | 4h | 最稳健 |

**推荐 A**：成本最低，收益最直接。1 epoch SFT 是 LLaVA-1.5 paper 标准。

## 4. 沿途发现的真实 bug

* **`return_logprob=True` 路径里 logprob 字段返回 NaN**：SGLang 的某个 sampler 路径在 Kimi Linear（KDA + MLA mix）下返回 NaN logprob 而非真实概率。LM-only 也复现。是 SGLang 上游问题，不是我们代码。

## 总结

之前我说 "16% epoch / 严重欠训" 是错的，按 epoch 算是 51%。但即便如此 1 epoch 才是 LLaVA-1.5 标准，**继续训到 step-2344 就达标**。`!!!!` 是 NaN logits 的可见症状（token 0 = `!` 巧合），续训能很大概率消除。
