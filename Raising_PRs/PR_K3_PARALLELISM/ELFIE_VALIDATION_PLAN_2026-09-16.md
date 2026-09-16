# Kimi K3 on the integration tree: what to run on GB200 / GB300 (2026-09-16)

For Elfie. Replaces `ELFIE_VALIDATION_PLAN_2026-09-10.md` (its branch heads, PR 4500 and PR 4412 are gone). Everything below is on `QIU023/torchtitan`, on top of upstream main `810e62786` or on the open PR bases named per item. Listed are only the components this tree owns; TP (#4499), quantile balancing (#4577), text-side CP (#4639) and Muon are the maintainers' PRs and not listed. Two parts: (1) what only a multi-node GB200 / GB300 box can measure, in the order it unblocks PRs; (2) what a single node adds at real width.

## 0. Setup

- Tree: `k3_on_4025` = `45ed0e3c1`, tag `k3_int_20260916c`, main `810e62786` + 72 commits: every component below at once. Use it for the combined cells and for anything not yet filed.
- PR branches (heads of 2026-09-16), for per-PR cells: `k3_pp_text` `de6f29514` (#4312, on an older main; rebase waits for the review), `k3_pp_mm` `384d576dc` (#4381, DEP, on #4312), `k3_cp_mm` `774e0b9b5` (#4380, dynamic vision CP, on #4639 `e06dcbee3`), `k3_ac_reuse_attention` `7ec30e5a9` (#4656, on main tip), `k3_moonep_seam` (#4751; its branch has one `moe.py` conflict against main tip, the tree carries the merged version), `pp_offload_review1` `d49bb388b` and `pp_balance_review1` `080f44208` (the two drafts stacked on #4312, filed as `k3_pp_offload` / `k3_pp_balance`), `lora_review2` (#4576).
- Environment: a torch nightly with the BFX9 kernels (main after #4484 needs it on Blackwell), Attention Gym from `meta-pytorch/attention-gym` main (what #4639's `pyproject.toml` pins), `spmd_types` and `torch_remat` as `pyproject.toml` pins. GB200 is SM100, so Attention Gym runs its CuTe KDA kernels; the one-node numbers below come from portable Triton kernels on SM120 and are not expected to agree bitwise across boxes. What is expected bitwise is stated per item.
- Protocol: `--debug.seed 42 --debug.deterministic --metrics.log_freq 1`, the 33-layer `kimi_k3_debugmodel`, 4096 tokens per step in 256-token micro-batches per dp rank (`--training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256`) unless stated. Report loss and grad norm at steps 1, 3, 10 per cell, and peak memory per rank; "bitwise" means identical printed values. Pair only cells with the same dp degree (the loader shards documents by dp rank).
- One-node holdings quoted below were measured on `45ed0e3c1` itself (the seeded 21-cell matrix `mx4_int0916c_0916_094907`, `phase13_k3like_48b_posttrain/K3_INT_20260916.md`, last section); every step-1 value equals the morning head's, so the earlier tables of that file read the same.

Template:

```
torchrun --nnodes N --nproc_per_node 8 -m torchtitan.train --module kimi_k3 --config kimi_k3_debugmodel \
  --debug.seed 42 --debug.deterministic --metrics.log_freq 1 --training.steps 10 \
  --training.num-tokens-per-train-step 4096 --training.num-tokens-per-microbatch-per-dp-rank 256 \
  <parallelism flags of the cell>
```

Recipe cells use `--module torchtitan_recipes.tests.b200 --config <recipe>` instead of the flavor.

## 1. What only a multi-node box can measure

### 1.1 The five axes together (the RFC's last row)

One node holds every 3-of-4 cell of the RFC matrix, including the three with CP that #4639 refused on its own head (fsdp2 x cp2 x tp2 `12.52262`, fsdp2 x cp2 x pp2 `12.52668`, tp2 x cp2 x pp2 `12.55824` at step 1; the last one bitwise with cp2 x tp2). What no node of eight holds is fsdp x tp x ep x pp x cp at once. On two nodes: `--parallelism.data_parallel_shard_degree 2 --parallelism.tensor_parallel_degree 2 --parallelism.context_parallel_degree 2 --parallelism.pipeline_parallel_degree 2 --parallelism.expert_parallel_degree 2 --parallelism.no-enable-sequence-parallel --parallelism.num_pp_microbatches 16` on the cp2 recipe (`kimi_k3_debugmodel_mm_allgather_kv_cp2`); on four nodes, the same with pp4. Report steps 1 / 3 / 10 and per-rank peak; expected: step 1 bitwise with the same cell at pp1 on one node (the pipeline changes nothing at step 1 in every pairing held so far).

### 1.2 Pipeline parallelism across nodes (#4312)

The two-node PP8 hang under 1F1B you found on the old tree, on the new stage class. `k3_pp_text`: pp8 1F1B with 8 micro-batches, once with and once without `TORCHTITAN_PIPELINE_NEIGHBOR_P2P=1` (your transport isolation, opt-in), and the pp8 x vp4 recipe (`kimi_k3_debugmodel_pp8_vp4`, Interleaved1F1B). Report hang or not (with `NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` for the first minute of a hang) and step 1 against a one-node run of the same command. Expected: step 1 identical between the switch on and off and identical to one node; one node holds pp2, pp8 1F1B and pp8 x vp4 bitwise with dp1 (`12.52917`).

### 1.3 MoonEP (#4751)

The first box that can run it: MoonEP's buffer asserts NVLink SHARP multicast (Hopper or newer behind an NVSwitch), which no box here has. `matrix_scripts/moonep_multicast_probe.py` reads the two flags first. Then `phase13_k3like_48b_posttrain/MOONEP_TEST_PLAN_2026-09-16.md`, cells 1 to 8, in short: the package's own tests on 2 and 4 ranks; ep4 x fsdp4 with the standard dispatcher as the reference; the same with `moe_comm_backend="moonep"` at `B = E / R = 8` slots (loss, the per-rank routed token count per layer, which the report says is equal across ranks, slot occupancy, step time, allocator statistics); the step-1 per-parameter gradient comparison between the two (non-expert weights bitwise, expert weights inside the standard dispatcher's own floor under another reduction order); a forced-hot router for both; `B = E / R - 1` and `B = 1` under it (the failure must name the bound, not hang); ep2 on two GPUs as the CI-sized cell. On the tree the expert side (`moon_ep_experts.py`, the `[E + B]` tables, prefetch, slot-gradient reduce) has run only through an in-process CPU double, so the first GPU run is a porting pass before a measurement.

### 1.4 PP rank-store offload and PP activation balance (drafts on #4312)

`attn_res_cache_offload` (the blocks a rank keeps for its later stages parked on pinned host memory) and `pp_balance` (saved activations of the heavy PP ranks parked on a light rank's GPU through the Mooncake Transfer Engine, RDMA where an HCA exists, TCP otherwise). Both are switches on `pipeline_kimi_k3` passed through `functools.partial` as a recipe's `pipelining_fn`; the balance knobs are `PPBalanceKnobs(pp_balance_source_ranks=..., pp_balance_dest_rank=..., pp_balance_min_tensor_mib=...)`. At debug width the logits dominate the peak and nothing shows; one node holds the seeded pp2 x vp2 pair bitwise for each switch (`12.53584` / `6.60274` / `3.21885`, rank 0 parking 960 tensors, 960 MiB a run, over TCP). What only your box gives: per-rank peak memory with and without each switch at the released width, the RDMA path of the Transfer Engine (never run here; `nvidia-cuda-runtime-cu12` next to a cu13 torch for the mooncake wheel), and `MC_INTRANODE_NVLINK=1` for same-node GPU peer transfers.

### 1.5 DEP hiding rate at a size that can show it (#4381)

`k3_pp_mm`: `kimi_k3_debugmodel_pp8_vp4_vit_dep` (the tower and the embedding on a stage of their own) with `vit_bubble` or `vit_prefetch`. At debug size the tower is too cheap to hide; run with at least 32 micro-batches and a real tower cost ratio (the released 27-layer tower against the text stages), read the bubble plan's log lines (planned encodes in bubbles / synchronous / idle slots), criteria fixed beforehand (backward hidden at or above 90%, forward at or above 25%). Numerics: step 1 bitwise with the same recipe without `vit_dep` (one node holds it at debug size).

### 1.6 Dynamic vision CP on real images and video (#4380)

`k3_cp_mm` or the tree: with `context_parallel_degree > 1` every image at or above `dynamic_cp_min_patches` (256) is split over a sub-CP group, smaller ones are replicated (log line "Dynamic CP: N large image(s) ... over M sub-CP group(s)"). Debug data has one 192-patch image per step, so the balancing case (several large images per step, sub-groups of 2 and 4) and video (bands are the same rows of every frame) need your data. One node holds the cut tower against the replicated tower at fp32 to 1e-6 forward and 1e-5 gradients on one image, a padded image, a two-frame video and a mixed batch, and the CP x TP and CP x PP cells with the tower cut (`tp2_cp2_pp2` bitwise with `cp2_tp2` at step 1).

## 2. What a single node adds at real width

### 2.1 AttnRes math recomputed in backward (#4656)

`k3_ac_reuse_attention` `7ec30e5a9` against its base `810e62786`, AC mode none, the same seed and batch: loss expected identical on every step (one node holds AC none / selective / full identical, `12.62200` / `10.81623` / `8.22700`); report peak memory and step time at a width where the residual math's activations matter (the debug width reads 14.17 GiB against 12.5 GiB under AC).

### 2.2 `torch.compile` of the decoder blocks and the tower (not filed)

`--compile.enable --compile.components model` on the tree: compiled against eager on the same seed, steps 1 / 3 / 10, step time after the first step, and the recompile count. One node holds the compiled dp1 cell running; the compiled-vs-eager gap is not yet located below module level, so a per-layer step-1 gradient comparison from your side is the useful artifact.

### 2.3 QAT accuracy at size

`kimi_k3_debugmodel_mx_qat` (MXFP4 weights, MXFP8 activations, fake-quant on every `GroupedExperts`, which is what the released index quantises). Your own suggestion from August; the one-node attribution on the debug model is 0.47 nat from the weights and 0.24 nat from the activations at step 1, a study at size is yours.

### 2.4 LoRA and QLoRA (#4576)

`kimi_k3_debugmodel_lora`, `kimi_k3_debugmodel_qlora_mxfp4` (packed MXFP4 frozen bases, NF4 as the other base): peak memory against full fine-tuning at the released width, and the adapter export round trip in the released key spelling.

### 2.5 Released weights: the step-0 anchor, then N steps

The debug flavor cannot anchor correctness; the released checkpoint can. Load it into the `Kimi-K3` flavor through the state-dict adapter (the released log-decay layout and the packed MXFP4 experts through torch's quantized Hugging Face reader, both on the tree), take the step-0 loss on a fixed batch against a vLLM or HF forward on the same batch (equal to bf16 rounding of the logits is the bar), then FSDP x TP x EP (x PP) at the released shape for N steps with per-rank peak. One node holds only the debug-size released-layout artifact (0 missing / 0 extra keys, bitwise export round trip).

### 2.6 Multi-token prediction layers

`kimi_k3_debugmodel_mtp`: the MTP loss terms next to the main loss at size, and the step-time cost of the extra layers. The released config ships MTP off; the layers exist for the report's training setup.

### 2.7 The Kimi-Linear-48B graft as a correctness anchor

Branch `k3_linear_graft` (not on the tree): Kimi-Linear-48B's released weights loaded with Attention Residuals at alpha 0 is numerically the original model at step 0, so a PP or CP cell at 48B width has an exact reference forward. The cleanest real-width anchor for the pipeline stage class and the CP routing, if a 48B run fits your time.

### 2.8 Optimizer state on resume, at scale

Your second finding on #4281. On the tree layer 0 has no attention-residual parameter, and main materialises optimizer state for every trainable parameter before save and load; one node holds an exact round trip for weights, Adam moments, step and learning rate. One check at your scale: save at step 5, resume, compare step 6 with an uninterrupted run (checkpoint contents identical; step 6 inside the run-to-run class of your box).

## 3. What to send back per cell

The command, the environment (torch build, Attention Gym commit, NCCL version), steps 1 / 3 / 10 loss and grad norm on rank 0, peak memory per rank, and for anything that hangs the NCCL INIT / NET log of the first minute.
