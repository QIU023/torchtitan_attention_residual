# verl rollout-vs-actor log-prob parity on the new tree (2026-09-07)

Where the 0.78-nat gap between the vLLM rollout and the torchtitan actor came from, how it was
located, and what fixed it. Tree: torchtitan `k3_int_20260906` (`4f7c87fc9`, run worktree
`/tmp/wt_k3int_cp` with the `rl` flavor), verl `kimi_k3_integration_rebased` (`4b08b917` plus the
commits below), export `/root/models/kimi-k3-debug-nt` (12 layers, 11 MoE layers of 32 routed
experts, top-4, 163,840-token vocabulary), two 5060 Ti (fsdp2, rollout TP 1, one vLLM replica per
GPU, `load_format: dummy`).

## The metric that hid it

Every verl table in `K3_INT_20260906.md` reports `training/rollout_probs_diff_*`, which is
`|exp(rollout_logp) - exp(actor_logp)|` per response token. This policy is random-init: its entropy is
11.52 nats over a 163,840-token vocabulary (ln V = 12.01), so every per-token probability is about
$e^{-11.5} \approx 1e-5$ and the metric is bounded by that whatever the two engines compute. The values of
4e-4 (max) and 1e-5 (mean) say nothing about parity. `verl/utils/debug/metrics.py` now also reports
`training/rollout_logprobs_diff_max/mean`, the same difference on the log-prob scale; every reward
cell on the new tree showed a mean of 0.78 nats at step 1, live or frozen sync:

| cell (`grpo-k3-newtree-reward-*.log`) | logprobs_diff_mean | max | probs_diff_mean |
| --- | --- | --- | --- |
| reward, live sync | 0.782 | 5.20 | 9.8e-6 |
| reward, frozen sync | 0.780 | 5.17 | 9.8e-6 |
| `LOGP_MBS=1` (one sequence per micro-batch) | 0.780 | 4.62 | 9.8e-6 |
| prefix caching off | 0.785 | 4.74 | 9.8e-6 |
| explicit KDA document offsets (`cuseq`) | 0.780 | 4.84 | 9.8e-6 |
| top-16 export config (the template's value) | 0.806 | 5.39 | 1.0e-5 |

In the batch itself (`KIMI_GRPO_DUMP_LOGPROBS`, 160 responses of 256 tokens): actor mean log-prob
-11.934, rollout mean -11.525, std about 1.0 on both sides, correlation 0.57, 58 percent of the
response tokens more than 0.5 nats apart, uniformly from token 0. The 0.41-nat offset is Gibbs'
inequality at work -- the sampler's own tokens look likelier to the sampler than to any other
distribution, by KL(q||p) -- so the two engines were computing distributions a good nat apart.

## Locating it

Standalone probes of the same export (the `parity_*.py` scratch probes, summarised here) had put torchtitan and vLLM
0.08 to 0.13 nats apart on natural prompts, sampled continuations, batched decode and prefix
caching -- the size of the model's own bf16-vs-fp32 sensitivity (0.091, 4.7 percent of positions
over 0.5). So the 0.78 was verl-specific. The batch dump was extended with the prompt tokens
(`prompts` plus `prompt_lengths`, fetched from the TransferQueue) and 40 of its sequences were
rescored offline (`matrix_scripts/verl_parity/`: `rescore_tt.py`, `rescore_tt_variants.py`, `rescore_vllm.py`,
`vllm_decode_consistency.py`; 10,240 response tokens):

| pair | mean abs diff (nats) | > 0.5 | max |
| --- | --- | --- | --- |
| near-fp32 torchtitan reference vs actor `old_log_probs` | 0.091 | 5.0% | 3.38 |
| near-fp32 reference vs bf16 single-sequence torchtitan | 0.092 | 5.1% | 2.46 |
| near-fp32 reference vs rollout `rollout_log_probs` | 0.781 | 58.3% | 4.51 |
| actor `old_log_probs` vs bf16 single-sequence torchtitan | 0.110 | 6.4% | 3.33 |
| offline vLLM (disk load, rollout settings) vs bf16 torchtitan | 0.122 | 7.0% | 2.33 |
| rollout `rollout_log_probs` vs offline vLLM, same sequences | 0.783 | 58.2% | 4.46 |
| offline vLLM decode vs prefill (raw / processed logprobs / sleep mode) | 0.103 | 5.6% | 2.45 |

The actor is right (0.09 from the reference, inside the bf16 floor). The rollout is 0.78 from the
reference AND 0.78 from an offline vLLM with the same export and the same engine settings
(`enforce_eager`, chunked prefill 512, `max_num_seqs` 8, `max_model_len` 1024, prefix caching,
`logprobs_mode`, sleep mode, temperature 1, top-k -1, top-p 1), so the difference is inside verl's
server: its weights.

## The cause: half the experts never arrive

The synced tensors were dumped on both sides. What the trainer ships is right: the first sync's
HF-named tensors are bitwise the export (`KIMI_GRPO_DUMP_SYNC`, 101 of 101 keys) and the adapter's
`to_hf` of the whole export round-trips all 1,428 keys. What the rollout holds is not: after the
first sync (`KIMI_GRPO_DUMP_VLLM`, compared with a disk-loaded vLLM model, vLLM-internal names) 323 of
345 tensors are identical and the 22 that differ are `routed_experts.w13_weight` and
`routed_experts.w2_weight` of all 11 MoE layers -- replica 0 holds experts 0-15 and zeros for 16-31,
replica 1 holds 16-31 and zeros for 0-15.

Under fsdp2 (dp_shard 2) the grouped expert stack `w1_EFD` (and w2, w3) is a DTensor sharded on
its expert dimension, and torchtitan's `to_hf` (`_get_local_experts_weights` in
`torchtitan/models/utils.py`) names only the LOCAL experts of such a stack, with their global ids --
the DCP convention, where the checkpoint writer reassembles the ranks. verl's full-tensor sync
(`get_per_tensor_param`) converts first and gathers second (`full_tensor()` on whatever is still a
DTensor), so each rank ships its 16 experts to the replica it feeds and the replica's other 16 keep
the `load_format: dummy` zeros. The `KIMI_GRPO_DUMP_SYNC` check missed it because it compared the
keys PRESENT in the dump against the export instead of the export's key set against the dump (51
expert keys for layer 1, not 96). verl's EP path (`iter_per_tensor_params_ep`) gathers experts
after the conversion, but only under `ep_enabled` and only for `mlp.experts.` names, so it neither
ran here nor would match Kimi K3's `block_sparse_moe.experts.` keys. The sharded (LoRA merged) path
already sends expert stacks whole with a slot table (`_expert_stack_slots`) and is not affected.

The pipeline cells have the same class of hole one level up: under pp2 each rank holds its stage's
layers only, `to_hf` names only those, and each replica received one stage and kept zeros for the
other. Measured on the log-prob scale with the expert-stack fix alone, the pp2 cell sat at 0.884
nats (`grpo-k3-newtree-pp2-stackfix.log`, first run).

## The fix (verl engine)

`get_per_tensor_param` takes every sharded expert stack out of the state dict before `to_hf`
(detected through the adapter's `from_hf_map`, the same test the shard path uses), converts the rest
as before, and appends a generator that gathers each stack whole with `full_tensor()`, one stack at a
time, and lets `to_hf` split the plain tensor into all its experts (`_iter_expert_stacks`). Memory
is one full stack per rank at a time, as the EP path already pays. Unit test:
`tests/workers/test_torchtitan_engine_expert_stacks.py` (2 pass). The adapter-only LoRA half carries
no expert stacks and is skipped. Under a pipeline the generator is wrapped once more
(`_iter_pp_gathered`): the stages stream in order, the owning rank broadcasts each tensor to the
other pipeline ranks as it yields it, and every rank yields every stage's tensors, one in flight at
a time (two-process test in `tests/workers/test_torchtitan_engine_pp_sync.py`).

Result on the reward cell and the cp2 cell (cp folds into the fsdp mesh, so its experts were
sharded the same way). The cp2 cell also needs one sequence per micro-batch now
(`log_prob_micro_batch_size_per_gpu=1`, `actor.ppo_micro_batch_size_per_gpu=1`,
`ref.log_prob_micro_batch_size_per_gpu=1`): the KDA context-parallel path runs one document per
batch and, since the engine passes the packed stream's document offsets (`4b08b917`), it refuses a
packed micro-batch instead of running the recurrent state across the sequences as the 09-06 cp2 cell
did; the engine raises with that hint.

| cell | before: logprobs_diff_mean / max / probs corr | after (step 1) |
| --- | --- | --- |
| fsdp2 reward cell (`grpo-k3-newtree-reward-stackfix.log`) | 0.782 / 5.20 / 0.49 | 0.107 / 3.63 / 0.965; steps 2-3: 0.107 / 2.77 / 0.963 and 0.108 / 3.15 / 0.963 (live sync, score 0.867 to 0.871) |
| pp2 (`grpo-k3-newtree-pp2-stackfix.log`, budget 2048, micro-batch 2) | 0.884 / 4.59 with the expert fix alone (each replica held one stage) | 0.108 / 3.63 with the stage gather; steps 2-3: 0.108 / 3.39, 0.108 / 4.48 (134-160 s/step) |
| fsdp2 reward cell, param + optimizer offload on (`grpo-k3-newtree-reward-offload-stackfix.log`) | same half-expert replicas as offload off | 0.109 / 3.33 at step 1 (372 s/step) |
| LoRA, merged sync on the shard path (`grpo-k3-newtree-lora-stackfix.log`) | not affected (stacks go whole with a slot table) | 0.109 / 3.52 at step 1, 0.107 / 3.15 at step 2 after a LoRA update through the merged sync (256-365 s/step) |
| cp2, one sequence per micro-batch (`grpo-k3-newtree-cp2-stackfix.log`) | not measured before the metric existed; the 09-06 cell ran with half-expert replicas | 0.110 / 3.24 (steps 2-3: 0.108 / 3.07, 0.108 / 3.29; 800-1300 s/step at micro-batch 1) |

0.107 is the floor the offline probes give for this export (0.10 to 0.12 between the two engines,
0.09 between bf16 and near-fp32 torchtitan), and the probability correlation goes from 0.49 to 0.96.

## Side findings kept

- The export's `text_config.num_experts_per_token` carried the template's 16 while the model routes
  top-4; `export_rl_newtree.py` now writes it from the router's `top_k` and both exports were patched.
  It moved the gap by 0.02 nats only, because the rollout was missing half its experts either way.
- KDA document boundaries: the model takes explicit `cu_seqlens` (`4f7c87fc9`) and the engine passes
  them for packed micro-batches; deriving them from position restarts inside the model moved the mm
  dp1 step-3 loss (7.612 vs 7.491) and was rejected.
- The probability-scale metric stays in the tables as what verl prints, with the caveat above.

## The ladder after the fixes (2026-09-08)

Every cell judged by `training/rollout_logprobs_diff_mean` at steps 1 and 3 with the synthetic
reward as the sync instrument (the actor must move for steps 2-3 to test the sync of updated
weights; nothing is being trained). The offline cross-engine floor for this export is 0.10-0.12.

| cell | GPUs | flavor / knobs | step 1 | step 2 | step 3 | s/step | note |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fsdp2 x ep2 | 2 | `rl`, `EP_SIZE=2`, log-prob micro-batch 4 | 0.108 | 0.107 | 0.107 | 424-768 | the fork's EP branch referenced `iter_per_tensor_params_ep` without importing it (never exercised before); the expert-stack gather holds under the EP placements |
| tp2 (fsdp1) | 2 | `rl`, `TP_SIZE=2`, micro-batch 2 | 0.110 | 0.110 | 0.106 | 433-1393 | two engine gaps: sequence parallel needs the packed stream padded to a multiple of the TP degree (the CP padding now pads to lcm(cp multiple, tp)), and the head's logits come back vocab-sharded (loss-parallel layout) and are gathered over the tp group before the loss side |
| QAT fsdp2 (`kimi_k3_rl_mx_qat`: MXFP4 weights / MXFP8 activations fake-quant on the routed experts) | 2 | micro-batch 2 | 0.502 | 0.502 | 0.503 | 646-1365 | NOT at the floor: the sync ships the bf16 masters, so the rollout samples the bf16 policy while the actor scores with fake-quantized experts; the 0.5 nats is the quantization error of MXFP4 on this random-init export, measured offline below, and the fix is to ship dequant(quant(w)) for the QAT-converted experts |
| QAT fsdp2, the sync ships dequant(quant(w)) for the QAT experts (verl `3dfd0430`) | 2 | micro-batch 2 | 0.260 | 0.258 | 0.255 | 349-448 | offline on the same sequences: the fake-quant forward sits 0.476 nats from the bf16 forward (38 percent of tokens over 0.5) and 0.779 from the bf16 rollout, so the earlier 0.50 was the quantization error; with the quantized weights shipped the remaining 0.26 is the MXFP8 activation fake-quant on the expert inputs that a bf16 rollout does not apply (attribution run pending the GPUs) |
