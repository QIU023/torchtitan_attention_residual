# PR-CP 正文(按 CLAUDE.md 的 PR-text 规则:无小标题、无粗体结构、无证据表)

发之前确认:4025 已 merge、树已 rebase 到 main、58 格对应轴已跑过。

--- PASTE BEGIN ---

Adds context parallel to Kimi K3. Before this, parallelize_kimi_k3 raised NotImplementedError for it; after, cp2 and cp4 train on both the text-only and the multimodal debug flavors.

MLA trades the sharded axis for heads with one fused all-to-all, runs the attention backend unchanged on the full sequence for its head subset, and trades back. KDA cannot do that under its own algorithm, so it has two modes: kcp keeps the sequence sharded end to end, taking a fixed-size halo for the causal convolutions and prefix-scanning the delta-rule recurrence over rank-local state fragments, which is what the report describes in 5.1.2; ulysses trades axes like MLA does, with the convolutions restricted to a contiguous channel slice, which is exact because they are depthwise. Both are checked against the same layer run without CP at cp2 and cp4: max absolute difference 1e-6 on bf16 outputs. The all-to-all itself is checked separately for exact placement, exact round trip and correct backward, also at cp4 -- at world size 2 a wrong permutation is symmetric and passes.

One change outside the model folder. Decoder.Config grows cp_via_sharding_config, default True, and validate_cp_backend is called only when it is set. That function's own docstring says it is for "the models that declare CP in ShardingConfig", but the base update_from_config calls it for every decoder. KDA runs on fla triton kernels that never see a DTensor, so no ShardingConfig can reach them and the layer implements CP itself; without the flag it has no way to say so. Every model that is declarative keeps the check.

K3 then enforces its own precondition instead: context_parallel_load_balancer must be None. The default, "headtail", permutes tokens across ranks. Both algorithms here read the sequence as rank-ordered contiguous chunks -- the all-to-all reassembles it in rank order, the recurrence passes state from rank r to r+1 -- so a permutation breaks both with every shape still lining up.

Two things worth stating because they are not obvious from the diff. The masks a layer receives under CP have been cut for ring attention, local queries against global keys; Ulysses reassembles the whole sequence, so it rebuilds the whole causal mask. That is correct only because this model rejects sample packing, and the docstring says so. And the vision tower has to stay in every rank's autograd graph: a rank whose slice holds no image tokens consumes no embedding, so the tower gets no gradient there, FSDP skips that rank's reduce, and the ranks deadlock -- observed as a 300s watchdog with one rank still in reduce_scatter while the other had moved two collectives ahead.

    torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 \
      --config kimi_k3_debugmodel --parallelism.context_parallel_degree 2 \
      --parallelism.context_parallel_load_balancer None

--- PASTE END ---

## 不进正文的支撑

| 声称 | 证据 |
|---|---|
| KDA CP 两模式与非 CP 一致 | cp2/cp4,max_abs ~1e-6(bf16 ulp 量级) |
| all-to-all 正确 | 放置逐元素精确、round-trip 逐位、backward 梯度正确,cp2 + cp4 |
| 端到端 | text cp2 12.45262 -> 10.25607;mm cp2 12.45567 -> 11.37354;cp4 两臂 3 步全过 |
| 死锁的真实形态 | 300s watchdog,rank0 卡在 reduce_scatter(seq 884),rank1 已到 885/886 |

## 未包含

vit dynamic CP(`vit_cp_plan.py` 已在树上,未接线)。塔目前在每个 rank 上完整跑一遍,
再按本 rank 的 token 切片取用 —— 正确但有冗余计算。
