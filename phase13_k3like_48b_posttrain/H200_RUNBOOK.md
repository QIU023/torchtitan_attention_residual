# H200 execution runbook (PLAN 3c leg) — ready to run

What the 5090 box physically cannot do (all-gather bound on no-P2P
PCIe, ~5 min/step at 48B; no room for fp32 masters) but is code-ready.
Rent 8xH200 (NVLink) per PLAN 3c and run these. Everything below has
been validated at smaller scale on the 5090; only the hardware changes.

## 0. Setup (same as 5090)

```bash
# clone forks, checkout attention_residual_dev, submodules
# /venv/main: torch 2.12 + fla-core + torchao (titan)
# /venv/verl: verl 0.9 editable + our patch (post-training)
# download moonshotai/Kimi-Linear-48B-A3B-Base; drop tokenizer.json placeholder
```

## 1. 48B QLoRA SFT at speed (the NF4 memory win manifests here)

At 48B the base weights dominate, so NF4 cuts both memory and (on any
fabric) all-gather traffic ~4x; on NVLink this runs at real speed.

```bash
source /venv/verl/bin/activate
export PYTHONPATH=/workspace/torchtitan_attention_residual/torchtitan:$PYTHONPATH
VERL_TORCHTITAN_FLAVOR=kimi_linear_48b_block_attn_res_gated_lora \
torchrun --nproc_per_node=8 -m verl.trainer.sft_trainer \
  engine=torchtitan optim=torchtitan \
  engine.data_parallel_shard_size=8 engine.spmd_backend=default \
  model.path=$MODEL_DIR model.trust_remote_code=true \
  data.train_files=<gsm8k or seq-KD corpus> \
  data.train_batch_size=64 data.micro_batch_size_per_gpu=1 \
  data.max_length=2048 data.max_token_len_per_gpu=4096 \
  trainer.total_training_steps=200 'trainer.logger=[console]'
# NO +engine.offload_policy needed on H200 (141 GB/card holds sharded
# bf16 48B ~12 GB/card + activations).
```

**NF4 quantize order (adapter wiring).** titan's trainer runs
build -> parallelize(shard) -> checkpointer.load(); QLoRA wants the base
packed from the LOADED weights, not init noise. Two order-correct paths,
both now supported by `lora.py`:

- **Post-load hook (from-scratch / unsharded load):** build the plain
  backbone bf16, load, then `quantize_lora_bases(model)` BEFORE
  `fully_shard` (quantize-then-shard, torchtune order). Validated 5090
  (`qlora_sft_demo.py`, unit test `test_post_load_quantize_hook`). The
  QLoRA flavor must leave `lora_quantize_base=None` so build does not
  pack early.
- **Build-time NF4 + NF4-aware load (sharded 48B):** keep
  `lora_quantize_base="nf4"`; the base is an NF4 param before
  `checkpointer.load()` re-quantizes the bf16 checkpoint into it. This is
  the path the flavor above takes; confirm titan DCP loads a bf16 shard
  into an NF4 base at 48B here (the one step the no-P2P 5090 cannot time).
`merge_lora_state_dict` folds the trained adapter back for HF export
(`to_hf` drops lora_* keys) regardless of which path trained it.

## 2. 48B full-param SFT + small GRPO (the H200 hard requirement)

fp32 masters (96 W + 96 grad + 384 AdamW = ~576 GB -> 72 GB/card) fit
H200, not 5090. Drop VERL_TORCHTITAN_FLAVOR (full-param baseline flavor)
and raise lr per PLAN. GRPO additionally needs the sglang AttnRes
overlay stood up (rollout server) -- see GRPO_STATUS.

## 3. Full-5D simultaneous composition validation

The one axis-combination the 5090 couldn't run (all of FSDP+TP+EP+PP
> 1 at once). Validated individually + at EP@896 on 5090; run the
simultaneous stack here:

```bash
torchrun --nproc_per_node=8 -m torchtitan.train \
  --module kimi_k3 --config kimi_linear_48b_block_attn_res \
  --parallelism.data_parallel_shard_degree 2 \
  --parallelism.tensor_parallel_degree 2 \
  --parallelism.expert_parallel_degree 2 \
  --parallelism.pipeline_parallel_degree 2 \
  --training.steps 50 --training.seq_len 1024 --checkpoint.no-enable
```

## 4. alpha trainable-vs-frozen A/B (HANDOFF 8 item 3)

On real 48B weights (not the 5090's random fixture): SFT the gated graft
with alpha frozen at 0 (== plain backbone) vs alpha trainable; the
trainable arm should diverge and (with a real task) win. Real evidence
the graft activates -- the 5090 could only prove step-0 identity.

## 5. 7.27 day-one (when weights + config + report drop)

1. Watch vLLM PR queue (config truth may precede the report).
2. Artifact-discovery checklist (K3_RELEASE_IMPACT 4): config.json
   field inventory, safetensors index dtype/shape spot-checks, base vs
   post-trained variants, license, chat template. The 34 official
   Kimi-Linear key patterns are already catalogued (state_dict_adapter).
3. **packed-MXFP4 import**: flip state_dict_adapter's from_quantized
   guard to a real unpack (torchao mx dequant, block 32). Never treat
   packed weights as plain tensors -- the guard already enforces this.
4. Regenerate flavors from the exact config (parameterized generator:
   N, ratio, experts, dims as variables); reconcile Gated-MLA form,
   Quantile Balancing, Per-Head Muon variant against the report.
5. Rerun smoke + parity; official-weight mapping; K3 model PR.

## Provisional pieces to reconcile at 7.27

Gated MLA gate form, Quantile Balancing exact rule, Per-Head Muon
variant, 2.8T dims, AttnRes N/placement, MXFP4-QAT recipe. Each has an
extension point / config knob ready (see K3_REPRODUCTION_STACK_STATUS).
