# K3 stack demo runbook — reproducible commands (2026-07-20)

Every working demo on the 8x5090 box, copy-paste. Two venvs: `/venv/main`
(titan) and `/venv/verl` (veRL 0.9 editable + our patch). Fork on
PYTHONPATH. Fixtures under /workspace/fake_hf (see git log of the
fixture-export commands). Box facts + perf caveats:
PERF_48B_ALLGATHER_INVESTIGATION.md. Component status:
K3_REPRODUCTION_STACK_STATUS.md.

## 0. Env

```bash
# titan
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
export PYTHONPATH=/workspace/torchtitan_attention_residual/torchtitan
# veRL (separate venv; DO NOT mix -- protects torch 2.12)
source /venv/verl/bin/activate
export PYTHONPATH=/workspace/torchtitan_attention_residual/torchtitan:$PYTHONPATH
```

## 1. Tests (titan)

```bash
cd $PYTHONPATH
pytest torchtitan/experiments/kimi_k3/tests/ -q          # 64 passed
PYTHONPATH="$PWD/..:$PYTHONPATH" pytest \
  ../phase3_attnres_pp_integration/dense_carrier/ -q      # 70 passed
```

## 2. Train smokes (titan)

```bash
# kimi 194m 5-step (fla triton path)
torchrun --nproc_per_node=1 -m torchtitan.train \
  --module kimi_k3 --config kimi_linear_194m_block_attn_res \
  --training.steps 5 --training.local_batch_size 2 \
  --training.global_batch_size 2 --training.seq_len 1024

# debugmodel CI flavor (seconds)
torchrun --nproc_per_node=1 -m torchtitan.train \
  --module kimi_k3 --config kimi_k3_debugmodel --checkpoint.no-enable
```

## 3. Parallelism

```bash
# PP pressure (naive vs adapter): phase3 launchers
STEPS=100 bash ../phase3_attnres_pp_integration/launch_4gpu_naive.sh
STEPS=100 bash ../phase3_attnres_pp_integration/launch_4gpu_adapter.sh

# TP=2 / EP=2 447m smokes
torchrun --nproc_per_node=2 -m torchtitan.train --module kimi_k3 \
  --config kimi_linear_447m_aligned_block_attn_res_n4 \
  --training.steps 50 --training.seq_len 512 \
  --parallelism.tensor_parallel_degree 2 --checkpoint.no-enable

# EP@896 (K3 expert count) construction+forward smoke
torchrun --nproc_per_node=8 \
  ../phase3_attnres_pp_integration/ep896_construction_smoke.py
```

## 4. 48B real weights (titan)

```bash
MODEL_DIR=/workspace/.hf_home/hub/models--moonshotai--Kimi-Linear-48B-A3B-Base/snapshots/3b171c17bfc4ee348599b6781a2ca8715c21c8dc
# bit-exact sharded load verification (4/4 EXACT-MATCH)
torchrun --nproc_per_node=8 verify_48b_load.py "$MODEL_DIR"
# alpha-graft step-0 identity (max|dlogit|=0.0):
for P in export baseline graft; do
  torchrun --nproc_per_node=8 verify_48b_graft_step0.py $P "$MODEL_DIR"
done
```

## 5. Post-training (veRL venv)

```bash
# full-param SFT 194m
VERL_TORCHTITAN_FLAVOR unset; torchrun --nproc_per_node=8 \
  -m verl.trainer.sft_trainer engine=torchtitan optim=torchtitan \
  engine.data_parallel_shard_size=8 engine.spmd_backend=default \
  model.path=/workspace/fake_hf/kimi_linear_194m model.trust_remote_code=true \
  data.train_files=/workspace/fake_hf/gsm8k_sft_2k.parquet \
  data.train_batch_size=16 data.micro_batch_size_per_gpu=2 \
  data.max_length=768 data.max_token_len_per_gpu=1536 optim.lr=1e-5 \
  trainer.total_training_steps=40 'trainer.logger=[console]'

# LoRA graft SFT: add
#   VERL_TORCHTITAN_FLAVOR=kimi_linear_194m_block_attn_res_gated_lora
# 48B LoRA SFT: model.path=$MODEL_DIR,
#   VERL_TORCHTITAN_FLAVOR=kimi_linear_48b_block_attn_res_gated_lora,
#   +engine.offload_policy=true (5090; ~5min/step -- see perf doc)
```

## 6. QLoRA + MXFP4 (titan, standalone probes)

```bash
# QLoRA SFT loop (bf16-LoRA vs NF4), FSDP-8
torchrun --nproc_per_node=8 <scratch>/qlora_sft.py
# MXFP4/MXFP8 QAT wrapper test
pytest torchtitan/experiments/kimi_k3/tests/test_mxfp4_qat.py -q
```

## Known-not-working (honest)

- GRPO: veRL v1 RL needs vllm/sglang/trtllm server; no hf rollout in RL.
- 48B full-param / QLoRA-in-trainer at speed: H200 (perf doc).
