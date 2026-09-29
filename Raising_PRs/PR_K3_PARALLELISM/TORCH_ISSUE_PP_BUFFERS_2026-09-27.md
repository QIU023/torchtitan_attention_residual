# torch issue 草稿：pipelining 的 send 到 step 末才 wait（2026-09-27 起草，09-29 改成只提 send）

- **09-29 改动：** torch#196463（"[pipelining] Allocate receive buffers just in time"，09-22 合入，09-23 以后的 nightly 都有）已经把接收缓冲改成收之前才分配、计算接手就交出。原来 issue 的第 1 个行为（接收缓冲整轮常驻）已由 torch 解决，只剩 send 这一半。PR A 同时去掉了接收覆盖（`PR_BODY_PP_CACHE_OPT_v3_2026-09-29.md`）。
- **粘贴区不放 5060 的数字：** 原来的 Measurement 一节是 5060 上的测量，按规则拿掉了，只留两卡复现（演示的是行为，不是性能数字）。H100 上 PR A 的对比出来以后，要不要补一段数字，你定。
- **提交位置：** pytorch/pytorch 的 issue，标签 `oncall: distributed`、`module: pipelining`。机器上没有 `gh`，需要你在网页上开。开好后把号填进 PR A 正文的 `<torch issue link>`。

# Title: [pipelining] The action-list runtime waits every send at the end of the step, keeping sent tensors alive

--- PASTE BEGIN ---

### Summary

`_PipelineScheduleRuntime` waits every send `Work` after the step's last action, and a pending `Work` keeps the sent tensor, and the whole storage of a sent view, allocated until then. A stage that has already handed an activation or a gradient to its peer therefore keeps it for the rest of the step, for any model. Receive buffers no longer have the analogous problem since #196463 allocates them just in time.

### Reproduction

Rank 0 sends a 256 MiB tensor with `batch_isend_irecv` and drops its reference; three seconds after rank 1 has received it, rank 0 still holds 256 MiB, and the memory is released only at `wait()`.

```python
import time, torch, torch.distributed as dist
dist.init_process_group("nccl")
r = dist.get_rank(); torch.cuda.set_device(r); dev = torch.device("cuda", r)
if r == 0:
    t = torch.randn(64, 1024, 1024, device=dev)
    works = dist.batch_isend_irecv([dist.P2POp(dist.isend, t, 1)])
    del t
    torch.cuda.synchronize(); time.sleep(3)
    print("before wait", torch.cuda.memory_allocated() / 2**20, "MiB")
    for w in works:
        w.wait()
    torch.cuda.synchronize()
    print("after wait", torch.cuda.memory_allocated() / 2**20, "MiB")
else:
    buf = torch.empty(64, 1024, 1024, device=dev)
    for w in dist.batch_isend_irecv([dist.P2POp(dist.irecv, buf, 0)]):
        w.wait()
dist.barrier(); dist.destroy_process_group()
```

### Proposal

- A forward send of stage s, micro-batch m can be waited when stage s starts its backward of m, since the receiver has used the tensor by then.
- An input gradient send of stage s, micro-batch m can be waited at the first later forward on the same rank whose input the receiving rank produced after the backward that consumed the gradient. The point comes from `pipeline_order`; a send without such a point keeps today's end of step wait.
- Single-stage schedules fuse sends with receives and would keep their current behavior.

Both waits are implemented today as overrides in a model's `PipelineStage` subclass in torchtitan (Kimi K3). I can send a PR if the direction is acceptable.

--- PASTE END ---
