# Paste-ready body for pytorch/pytorch

Title: `[pipelining] warm up P2P group communicators eagerly in STATIC inference mode`

---

Before: in STATIC inference mode, the group communicator behind a pipeline edge
is created lazily by the first mixed `_batch_p2p` call, which is 1F1B steady
state. After: `_warmup_p2p` runs `_get_init_p2p_neighbors_ops` through
`_batch_p2p` once after the vote, gated by `not p2p_done`, so every edge
communicator is bootstrapped at initialization. This implements the TODO that
`_warmup_p2p` already carries, with the fix it prescribes.

The lazy path hangs multi-node pipelines. On a 2-node PP8 run (one stage per
rank, 1F1B, static stage metadata), all stages start and then ranks 6/7 block
300 s in NCCL communicator creation for edges 5->6 and 6->7 until the
collective timeout kills the run. In steady state a rank cannot reach the op
that would create edge (r, r+1) before its earlier recvs complete, so
bootstraps serialize behind schedule dependencies; across nodes each bootstrap
is a network rendezvous measured in seconds, and the tail edges accumulate past
the timeout. Single-node runs never show it because shm/NVLink bootstraps are
milliseconds, three orders of magnitude below the timeout. DYNAMIC mode never
shows it because `_send_meta`/`_recv_meta` already touch every edge in
dependency order before the schedule starts -- this change gives STATIC mode
the same guarantee at the same point the non-PipelineStage legacy branch
already has it. The fake-process-group early return is unaffected.

Cost is one 1-float send/recv per edge per direction, once per process. Only
`torch.distributed.pipelining` executes this path; frameworks with their own
schedules (Megatron-LM, DeepSpeed) never reach it. Single-node PP8, 8 ranks,
deterministic mode: losses bitwise identical with and without the patch, 2
steps. The 2-node hang needs two nodes to reproduce; validation on the cluster
that showed it is pending.
