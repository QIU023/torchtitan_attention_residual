# 4312：Tianyu 对 `stage_class` 的裁决，以及数值那条还没贴（2026-09-11 09:13 之后）

## 1. 数值回复**没有发出去**

`parallelize.py:267`（id `3985333653`，"I'm not convinced that the numerics gap can be this large…
How do you prove it's not caused by bugs?"）这条线程下面**只有他 02:12 的提问，没有任何回复**。
06:31–09:13 之间发出去的 13 条覆盖了其他每一条评论，唯独漏了这一条——而这正是他两轮都在问的那条。

`REPLY_4312_NUMERICS_2026-09-11.md` 的 PASTE 块还在本地。发之前按
`phase13_k3like_48b_posttrain/PP_NUMERICS_VERDICT_2026-09-11.md` 的五条改：删掉 50/100 步两列、
把"we are not offering a mechanism"改写成噪声带、step-1 的 ulp 自我更正 + 机制（栈在 stage 边界由
`torch.stack` 重建而不是 `torch.cat` 长出来，布局变了 → fp32 matmul 的归约顺序可能变）、
底行给 3–5 个重排的带宽、以及把 24 层的 step-1 逐参数剖面放到最前面。

## 2. 他对 `stage_class` 给了具体做法（09:11，`parallelize.py:247`）

> For now could you reconstruct an `AttnResPipelineStage` using the fields from the constructed stage.
> ```
> stage = AttnResPipelineStage(model_chunk, stage_idx, num_stages, device,
>                              group=pp_mesh.get_group("pp"), get_mesh=get_mesh, **stage_kwargs)
> ```
> … the tradeoff is to have clean integration point without intrusive change to existing
> `pipeline_parallel.py`. We need to take some time and see if this is the proper abstraction for attn res.

意思很明确：**把 `pipeline_llm` 的 `stage_class` 参数拿掉**，改为在 K3 的 pipelining 入口里，
用 core 已经构造好的 stage 的字段**重建**成 `AttnResPipelineStage`。他宁愿要"模型侧有点 hacky"，
也不要 core 多一个参数——并且明说抽象层面的结论要以后再定。

### 可行，核对过的实现形状

`pipeline_llm` 返回之后，schedule 里握着 core 构造的 `PipelineStage`；重建之后必须把它们换进
schedule，否则跑的还是旧对象。核对了 torch 的 `PipelineScheduleMulti.__init__`：

- `self._stages = stages`；`self._num_stages` / `pp_group_size` / `rank` 取自 `stages[0]`，
  重建后数值相同，不用动；
- 它会给每个 stage 写 `stage.stage_index_to_group_rank`，**这个要搬到新对象上**；
- `_stages_forward_initialized` / `_stages_backward_initialized` 构造时都是 `False`，
  也就是说**只要在第一次 `step()` 之前换掉就没有遗留状态**（pipelining_fn 正好在 setup 期）；
- `_should_compute_loss = lambda stage: stage.is_last and has_loss` 以 stage 为入参，无所谓；
- `pipeline_order` 按 stage 下标算，无所谓。

`get_mesh` 没有以原样存在 stage 上，而是 `self._mesh_cache = _MeshCache(get_mesh_cb=get_mesh)`，
所以最省事的是**把整个 `_mesh_cache` 搬过去**，不必去拿那个回调。

```python
pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(model, **kwargs)
built = _schedule_stages(pp_schedule)          # core 构造的 PipelineStage
stages = []
for s in built:
    a = AttnResPipelineStage(s.submod, s.stage_index, s.num_stages, s.device, group=s.group)
    a._mesh_cache = s._mesh_cache              # core 装进去的 get_mesh 回调
    a.stage_index_to_group_rank = s.stage_index_to_group_rank   # schedule 构造时写的
    stages.append(a)
if isinstance(pp_schedule, PipelineScheduleSingle):
    pp_schedule._stage = stages[0]
else:
    pp_schedule._stages = stages
```

`s.submod` 在 `pipeline_llm` 里已经被 `parallelize_fn` 换成并行化后的模块，所以重建拿到的是对的；
`has_first_stage` / `has_last_stage` 由 `stage_index` 与 `num_stages` 推出，重建后相同。

### 代价，要如实写给他

1. 从"core 多一个参数（+4/−1，默认不变）"换成**碰 torch 的两个私有属性**
   （`stage._mesh_cache`、`schedule._stage` / `_stages`）。他自己说了这是 trade-off，接受。
2. core 先构造一批 `PipelineStage` 再被丢掉——多一次构造，不影响正确性。
3. **版本漂移的风险**：如果 torch 以后在 schedule 构造时对 stage 多做一件事，我们的替换会静默漏掉；
   一个 core 参数不会有这个问题。这条说一次即可，不要争。

### 与他另一条的关系

他早前（`parallelize.py:184`）要的是"extend `pipeline_parallel.py#L144` 而不是写模型专用的 split"，
那条现在是给 #4560 的 `pipeline_with_first_stage_modules` 加 `last_stage_module_fqns`。
两条不冲突：**split 走 core 的公开入口，stage 类不进 core**。做完这两件事之后，
`pipeline_parallel.py` 在本 PR 里只剩 `last_stage_module_fqns` 那一处。

## 3. 已发出去的 13 条里，一处措辞要留意

`parallel_dims.py:566` 那条写了 "the fix is PyTorch's own `pipeline_per_direction_p2p`…, which our
stage inherits because it subclasses `PipelineStage`"。**成立，但有版本下限**：这个旋钮是
pytorch/pytorch#186173（落地提交 `420415fa0`，2026-06-16）加的，torch 2.13 里没有
（`torch.distributed.config.pipeline_per_direction_p2p` 不存在）。torchtitan 钉的是 nightly，
所以对 CI 成立；但 Elfie 的两节点环境要确认 torch 够新。他若追问，补一句版本下限即可。
