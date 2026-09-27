# torch issue 草稿：pipelining 的接收缓冲常驻、send 到 step 末才 wait（2026-09-27）

- **用途：** 计划 §6 的第一步。PR A（`pp_review_optimize` = `be9e2fa69`）在 `AttnResPipelineStage` 里覆盖了这两个行为，#4765 的正文要链接这个 issue。
- **提交位置：** pytorch/pytorch 的 issue，标签 `oncall: distributed`、`module: pipelining`。机器上没有 `gh`，需要你在网页上开。开好后告诉我 issue 号，我把它补进 #4765 正文的 `<torch issue link>`。
- **数据来源：**
  - `PP_OPTIMIZE_REPORT_2026-09-24.md` §2 和 §3：8 × RTX 5060 Ti，pp8 × vp2，Interleaved1F1B，16 个 micro-batch，seq 3584，FullAC；
  - `kit_pp_optimize_2026-09-24/stash_probe.py`。
  - 这是 09-24 的 base 上测的。issue 里只写 torch 的行为和这组测量本身，不当作任何 PR 的结果。

# Title: [pipelining] Receive buffers live for the whole run and send works are waited only at the end of the step

--- PASTE BEGIN ---

### Summary

Two behaviors of `torch.distributed.pipelining` hold every pipeline stage's activations and gradients longer than the schedule needs, for any model:

1. `_PipelineStageBase` allocates one receive buffer per micro-batch for every input and input gradient in `_setup_forward_recv_info` / `_setup_backward_recv_info`, and keeps them across steps.
2. `_PipelineScheduleRuntime` waits every send `Work` after the step's last action, and a pending `Work` keeps the sent tensor (and the storage of any view) allocated until then.

### Measurement

8 x RTX 5060 Ti, pp8 with two virtual stages per rank, `ScheduleInterleaved1F1B`, 16 micro-batches, 3584 tokens per micro-batch, full activation checkpointing, torch nightly 2.15.0.dev20260906:

- The receive buffers alone hold 2.41 to 4.16 GiB per rank between steps.
- Issuing each forward send from the stage and waiting it when the stage's backward of the same micro-batch starts lowers the per rank peak by 0.52 to 1.30 GiB.
- With both behaviors overridden (plus model side changes), loss and grad norm stay bitwise identical to the unmodified run on all 100 steps.

A two rank reproduction of the pinning: rank 0 sends a 256 MiB tensor with `batch_isend_irecv` and drops its reference; three seconds after rank 1 has received it, rank 0 still holds 256 MiB, and the memory is released only at `wait()`.

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

- Receive buffers: allocate each one when its receive is posted and drop the stage's reference once the stage has read it (`_retrieve_recv_activations`, `_retrieve_recv_grads`), optionally behind a flag.
- Send waits in `_PipelineScheduleRuntime`:
  - a forward send of stage s, micro-batch m can be waited when stage s starts its backward of m, since the receiver has used the tensor by then;
  - an input gradient send of stage s, micro-batch m can be waited at the first later forward on the same rank whose input the receiving rank produced after the backward that consumed the gradient; the point comes from `pipeline_order`, and a send without such a point keeps today's end of step wait.

Both are implemented today as overrides in a model's `PipelineStage` subclass in torchtitan (Kimi K3). I can send a PR for either if the direction is acceptable.

--- PASTE END ---
