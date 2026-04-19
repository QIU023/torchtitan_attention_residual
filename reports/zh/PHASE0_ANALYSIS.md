# Phase 0 调研报告：在 torchtitan 中实现 AttnRes — Go/No-Go 决策

**日期**：2026-04-12
**结论**：**GO — 选择 torchtitan 作为主框架**

---

## 1. 摘要

torchtitan 的 pipeline parallelism 基础设施支持 Block AttnRes 集成，**无需修改 PyTorch pipelining 库本身**。实现完全在 model 层面：修改 `Decoder.forward()` 让 block summaries 在 layers 之间流动，并在 PP 中间 stage 返回 tuple 输出。`PipelineStage` 原生支持通过 P2P send/recv 传递 tuple tensor。

---

## 2. 关键发现

### 2.1 Moonshot 官方 repo
- **没有可运行代码** — 只有 PDF 和 README 里的 pseudocode
- Pseudocode 足够清晰，可以直接对照实现（论文 Figure 2）
- 重要细节：`block_size` 计数的是 ATTN + MLP 两个 sublayer，每个 transformer layer 有两个 sublayer

### 2.2 PyTorch Pipelining API（`stage.py`）

**关键发现**：`PipelineStage` 原生支持 tuple 输出。

```python
# stage.py:31 — _normalize_model_output_as_tuple
output_tuple = output if type(output) is tuple else (output,)

# stage.py:433 — get_fwd_send_ops 遍历 tuple 的每个元素
for idx, out in enumerate(output_tuple):
    dst_stages = self.act_send_info[idx]
    for dst in dst_stages:
        ops.append(dist.P2POp(dist.isend, out, peer_global_rank, self.group))
```

**Activation 流向**：
1. `forward_one_chunk()`（line 660）：对非首 stage 调用 `_retrieve_recv_activations()`
2. 收到的 activations 作为 **positional args** 传给 `submod.forward()`
3. 如果 stage N-1 返回 `(tensor_a, tensor_b)`，stage N 的 forward 就会被调用为 `forward(tensor_a, tensor_b, **kwargs)`

**没有 hook 机制** — 没有 `pre_send`/`post_recv` 回调。但 tuple I/O 让我们不需要 hook 就能跑通基础方案。

参考：PyTorch RFC [#128665](https://github.com/pytorch/pytorch/issues/128665) 是一个还没被解决的 issue，专门在请求给 PipelineStage 加 send/recv 的用户控制点。这进一步确认了"走 tuple 输出"是当前最合理的工程路径。

### 2.3 torchtitan 模型结构

**Decoder.forward()**（`models/common/decoder.py:124`）：
```python
def forward(self, tokens, attention_masks=None, positions=None):
    h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
    for layer in self.layers.values():
        h = layer(h, self.freqs_cis, attention_masks, positions)
    h = self.norm(h) if self.norm is not None else h
    output = self.output(h) if self.output is not None else h
    return output
```

**PP stage 的处理方式**：`pipeline_module_split()` 深拷贝整个 model 再按 stage 裁剪掉不属于本 stage 的 layers。中间 stage 的 `tok_embeddings`、`norm`、`output` 全都是 `None`，靠 `if ... is not None` 的 passthrough 机制跳过。

**kwargs 传播**：`attention_masks` 和 `positions` 通过 `pp_schedule.step(**extra_inputs)` 以 kwargs 的形式传给**所有 stage**（见 `trainer.py:670`）。

### 2.4 Kimi infra 工程师的实现经验（Reku，知乎回答）

关键引用：

> "大致思路就是在 pipeline 并行的通信后增加一个适配器，将收到的 block 与适配器中缓存的 block 进行拼接。反向也是类似思路，会收到所有 block 的 grad，所有的 grad 就在适配器里面做累加，需要发到下一个 stage 的就直接把累加 buffer 发出去。整个代码逻辑还是比较对称的，不影响网络内部逻辑。"

> "在最常用的 interleave pipeline 的调度下，send/recv 的开销在 steady 阶段非常容易掩盖，只有 warmup 和 cooldown 阶段的一点点通信会暴露出来。"

> "cross-stage caching 会导致累加顺序的改变……这会导致一些 debug/精度对齐上的困难（比如 pipeline 并行配置变了，loss/norm 没法做到完全一致）"

> "在这个具有良好局部性的算法下，cross-stage caching 的通信优化就比较容易想到，除了第一个 vp chunk 外，后面的所有 pipeline 通信都是常数，解决了不对称的问题。"

这段话给了两个重要信号：
1. **adapter 模式**本身代码逻辑对称，不侵入 model 内部 —— 对 torchtitan 也适用
2. **精度对齐会痛** —— 改 PP 配置后 loss/norm 可能不完全一致，需要提前做好单卡对照组

---

## 3. 实现架构

### 3.1 模型层改动（Phase 2：单卡正确性）

**修改后的 forward 签名**：
```python
class Decoder(BaseModel):
    def forward(self, tokens, blocks=None, *, attention_masks=None, positions=None):
        h = self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens

        if self.use_attn_res:
            if blocks is None:
                blocks = [h]  # token embedding 作为第一个 "block"
            partial_block = h

            for layer in self.layers.values():
                blocks, partial_block = layer.forward_attn_res(
                    blocks, partial_block, self.freqs_cis, attention_masks, positions
                )

            # 输出前最后一次 cross-block attention
            h = block_attn_res(blocks, partial_block, self.final_attn_res_proj, self.final_attn_res_norm)
        else:
            for layer in self.layers.values():
                h = layer(h, self.freqs_cis, attention_masks, positions)

        h = self.norm(h) if self.norm is not None else h
        output = self.output(h) if self.output is not None else h

        # PP 中间 stage：把 blocks 一起返回
        if self.use_attn_res and self.output is None:
            blocks_tensor = torch.stack(blocks)  # [N, B, T, D]
            return (output, blocks_tensor)
        return output
```

**修改后的 TransformerBlock**（对照论文 pseudocode）：
```python
class Llama3TransformerBlock(TransformerBlock):
    def forward_attn_res(self, blocks, partial_block, freqs_cis, attention_masks, positions=None):
        # attention 之前做一次 block attnres
        h = block_attn_res(blocks, partial_block, self.attn_res_proj, self.attn_res_norm)

        # block 边界：开启新 block
        if self.layer_id % self.block_size == 0:
            blocks.append(partial_block)
            partial_block = None

        # Self-attention
        attn_out = self.attention(self.attention_norm(h), freqs_cis, attention_masks, positions)
        partial_block = partial_block + attn_out if partial_block is not None else attn_out

        # MLP 之前再做一次 block attnres
        h = block_attn_res(blocks, partial_block, self.mlp_res_proj, self.mlp_res_norm)

        # MLP
        mlp_out = self.feed_forward(self.ffn_norm(h))
        partial_block = partial_block + mlp_out

        return blocks, partial_block
```

**每个 layer 新增的参数**（增量非常小）：
- `attn_res_proj`：`nn.Linear(d_model, 1, bias=False)` — pseudo-query（**必须零初始化**）
- `mlp_res_proj`：`nn.Linear(d_model, 1, bias=False)` — pseudo-query（**必须零初始化**）
- `attn_res_norm`：`RMSNorm(d_model)`
- `mlp_res_norm`：`RMSNorm(d_model)`

**关键：零初始化**。论文 Section 5 明确说：pseudo-query 全部零初始化，这样初期 softmax 退化为均匀权重，等价于标准 residual，避免训练前期震荡。

**另一个重要的消融发现**：论文测试了 multihead AttnRes（H=16），结果 loss 反而更差（1.752 vs 1.746）。所以实现时就是单头 —— 每个 sublayer 一个标量 pseudo-query，不要做 multihead。

### 3.2 PP 集成（Phase 3：Naive 方案，不做 caching）

**和现有 PipelineStage 的配合方式**：

1. **Stage 0（首 stage）**：`forward(tokens, blocks=None, attention_masks=..., positions=...)`
   - 建立 `blocks = [tok_embedding]`
   - 处理 layers 0..K-1，累积 block summaries
   - 返回 `(partial_block, blocks_tensor)` —— 两个 tensor 的 tuple

2. **Stage 1..N-2（中间 stage）**：`forward(partial_block, blocks_tensor, attention_masks=..., positions=...)`
   - Positional arg 映射：`tokens=partial_block`，`blocks=blocks_tensor`
   - Unstack blocks，处理本 stage 的 layers，返回更新后的 `(partial_block, blocks_tensor)`

3. **Stage N-1（末 stage）**：和中间 stage 一样，但最后跑 final norm + output
   - 返回单个 logits tensor（不是 tuple）

**每次 stage 传递的通信量**：O(N × B × T × D) —— 发送所有 block summaries。这是不做 caching 的 naive 方案。

### 3.3 Cross-Stage Caching 优化（Phase 3+：进阶）

Kimi 工程师讲的 adapter 模式：
- 每个物理 stage 把前几个 virtual stage 收到的 blocks 缓存在本地
- 第一个 VP chunk 走完后，所有 blocks 已经分发完毕
- 之后的 VP chunk 只需要传增量 blocks（每次传输 O(P × Np × d) 而不是 O(C × Np × d)）

**在 torchtitan 里可行的实现路径**：
- **方案 A**：继承 `PipelineStage`，重写 `get_fwd_send_ops` / `get_fwd_recv_ops`
- **方案 B**：在 model 内部维护缓存 —— Decoder 持有一个 `_blocks_cache` dict，按 `(microbatch_id, virtual_stage_id)` 索引
- **方案 C**：往 PyTorch pipelining 上游提一个 hook API（长期方案）

**建议**：先做 naive 方案（Phase 3），caching 作为 Phase 3+ 的性能优化。Naive 方案是正确的而且能演示核心算法，caching 是性能优化，可以 profiling 之后逐步加上去。

---

## 4. 风险评估

| 风险 | 严重度 | 应对 |
|------|--------|------|
| Blocks tensor 跨 stage 尺寸不同 | 低 | PipelineStage 对每个 stage 分别推断 metadata，不同 stage 可以有不同的 output shape |
| Forward 签名变了，非 PP 路径被破坏 | 低 | `blocks=None` 默认值 + `use_attn_res=False` 保持原行为 |
| Stacked blocks 的梯度正确性 | 中 | 写充分的 unit test，单卡 vs fake PP 做数值对比 |
| torchtitan maintainer 不接受 model 层改动 | 中 | 按 CLAUDE.md 指南走，先放 experiment 目录，被接受后再进 core |
| Blocks_tensor 占太多 activation memory | 低 | 每 token O(N×d)，N~8，相对每层 activation 量几乎可忽略 |
| Pseudo-query 零初始化被 model converters 破坏 | 中 | 在 `init_states()` 里显式校验 |
| Cross-stage caching 改了梯度累加顺序 → 精度对齐难 | 中 | 提前准备好单卡 reference，对比时允许一个 tolerance |

---

## 5. 需要修改的文件清单

### 核心改动：
- `torchtitan/models/common/decoder.py` — Decoder.forward() 签名 + AttnRes 循环
- `torchtitan/models/llama3/model.py` — Llama3TransformerBlock 加 AttnRes 参数
- `torchtitan/models/llama3/config_registry.py` — 加 AttnRes config 选项

### 新文件：
- `torchtitan/models/common/attn_res.py` — `block_attn_res()` 函数 + `AttnResConfig`

### Config 改动：
- `torchtitan/config/` — 加 `--model.use_attn_res`、`--model.attn_res_num_blocks` 等 flag

### 测试：
- `tests/unit/models/test_attn_res.py` — Block AttnRes 正确性的 unit test
- `tests/integration_tests/` — 带 AttnRes 的 PP 集成测试

---

## 6. 工作量预估

| Phase | 时长 | 算力 | 成本 |
|-------|------|------|------|
| Phase 2：单卡正确性 | 5-7 天 | 1× 5090 | ~$40 |
| Phase 3：Naive PP（fake PP + 真 8 卡） | 7-10 天 | 1× 5090 + 8× 5090 | ~$280 |
| Phase 3+：Cross-stage caching | 3-5 天 | 8× 5090 | ~$130 |
| Phase 5：PR + Blog | 3-5 天 | — | — |
| **合计** | **~4 周** | | **~$450** |

> 价格基线（Vast.ai，2026-04-13 实时报价）：
>
> - **1× RTX 5090**（California，AMD EPYC 7742，PCIE 4.0 x16）：**$0.362/hr**，DLPerf 193.7，verified，reliability 95.4%
> - **8× RTX 5090**（Oregon，AMD EPYC 7K62，PCIE 4.0 x16，297 ports）：**$2.616/hr**，DLPerf 475.5，verified，reliability 98.5%
>
> 估算假设：debug-heavy 阶段按实际 GPU 占用 ~50% 计，训练阶段按连续占用计。

---

## 7. 下一步动作

1. **在 torchtitan 开 RFC issue** —— 描述方案、引论文、求 maintainer 反馈
2. **启动 Phase 2** —— 实现 `block_attn_res()`，写单卡正确性测试
3. **租一张 5090**（Vast.ai，~$0.362/hr）—— 做 loss curve 对齐验证，对标论文小规模点的 delta
