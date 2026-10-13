"""The 18-cell multimodal parallelism matrix on main (Python-config era), logbook kit only.

MM18_CELL: comma list of degrees, e.g. "dp=2,tp=2,ep=2,pp=2,cp=1" (missing keys are 1). Sequence parallelism is on
under TP unless CP is on (the K3 vision splice under CP refuses SP). MM18_ARM: base | lora | lora_mla (MLA projections only) | mxqat (Elfie's MX QAT alone) | lora_mxqat (LoRA over Elfie's MX QAT,
on the stack branch). MM18_SEED: the seed checkpoint folder of the arm. MM18_STEPS: steps (10).
Kimi K3 debug model (main's 17 layers), multimodal cc12m-test, seq 256 (512 at CP4), 4096 tokens per step in
one-sequence micro-batches per dp rank, 8 pipeline micro-batches, AdamW (DistMuon has no TP), seed 42, deterministic.
cell() loads the seed (model only); seed() writes it from a single-GPU build of the same arm.
"""

import os

from torchtitan.components.checkpointer import CheckpointManager
from torchtitan.components.optim import AdamW, OptimizersContainer
from torchtitan.config import ParallelismConfig


def _degrees(spec: str) -> dict[str, int]:
    degrees = {"dp": 1, "tp": 1, "ep": 1, "pp": 1, "cp": 1, "vpp": 1}
    for item in filter(None, spec.split(",")):
        key, value = item.split("=")
        degrees[key] = int(value)
    return degrees


def _build(degrees: dict[str, int]):
    from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

    cp, tp = degrees["cp"], degrees["tp"]
    # The recipe validates DistMuon against TP and attention against CP when it builds, so it is built at tp = cp = 1
    # and the two degrees are set afterwards, as the b200 and h100 recipes do.
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=degrees["dp"],
        expert_parallel_degree=degrees["ep"],
        pipeline_parallel_degree=degrees["pp"],
        enable_sequence_parallel=tp > 1 and cp == 1,
    )
    # CP flex attention needs the sequence divisible by cp * 128, so CP4 runs 512-token sequences.
    seq_len = max(256, 128 * cp)
    config = kimi_k3_debugmodel(seq_len=seq_len, parallelism=parallelism)
    config.parallelism.tensor_parallel_degree = tp
    config.parallelism.context_parallel_degree = cp
    if degrees["pp"] > 1:
        config.parallelism.num_pp_microbatches = 8
    if degrees["vpp"] == 2:
        # Two stages per rank is the interleaved schedule's default.
        config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    else:
        assert degrees["vpp"] == 1, degrees
    config.training.num_tokens_per_train_step = 4096
    config.training.num_tokens_per_microbatch_per_dp_rank = seq_len
    config.training.steps = int(os.environ.get("MM18_STEPS", "10"))
    config.metrics.log_freq = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    config.optim.optimizer = OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)])
    transforms = []
    if cp > 1:
        from torchtitan.config.transform import ContextParallelTransform
        from torchtitan.distributed.context_parallel import HeadTailCPLoadBalancer
        from torchtitan.models.common.attention import FlexInnerAttention
        from torchtitan.models.common.attention.cp_attention import KVAllGatherCPFlexInnerAttention
        from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
        from torchtitan.models.common.attention.kda import InnerKDA

        config.parallelism.context_parallel_load_balancer = HeadTailCPLoadBalancer.Config()
        transforms.append(
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: KVAllGatherCPFlexInnerAttention,
                    InnerKDA: ContextParallelInnerKDA,
                }
            )
        )
    arm = os.environ.get("MM18_ARM", "base")
    if arm == "lora":
        from torchtitan.config.transform import LoRATransform

        transforms.append(LoRATransform(rank=8, alpha=16.0))
    elif arm == "lora_mla":
        # LoRA on the MLA projections only: stages holding only KDA layers have nothing to train.
        from torchtitan.config.transform import LoRATransform

        mla = ["attention.wq_a", "attention.wq_b", "attention.wkv_a", "attention.wkv_b", "attention.wo"]
        transforms.append(LoRATransform(rank=8, alpha=16.0, target_modules=mla))
    elif arm in ("mxqat", "lora_mxqat"):
        # Elfie's MX QAT (stack branch only): the recipe's MXFP4 policy picks the weights; lora_mxqat then wraps them.
        from typing import cast

        from torchtitan.config.transform import LoRATransform, MXQATTransform
        from torchtitan.models.kimi_k3.model import KimiK3Model
        from torchtitan.models.kimi_k3.quantization import MXFP4_QUANTIZATION_CONFIG
        from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter
        from torchtitan.quantization.mx_qat.checkpoint import MXFP4CheckpointPolicy

        adapter = KimiK3StateDictAdapter(cast(KimiK3Model.Config, config.model), hf_assets_path=None)
        policy = MXFP4CheckpointPolicy.from_config(MXFP4_QUANTIZATION_CONFIG, adapter.hf_linear_weight_mapping())
        transforms.append(MXQATTransform.from_weight_fqns(config.model, adapter.qat_weight_fqns(policy)))
        if arm == "lora_mxqat":
            transforms.append(LoRATransform(rank=8, alpha=16.0))
    if transforms:
        from torchtitan.config.transform import apply_transforms

        config = apply_transforms(config, transforms)
    return config


def cell():
    config = _build(_degrees(os.environ["MM18_CELL"]))
    config.checkpointer = CheckpointManager.Config(
        interval=100000, initial_load_path=os.environ["MM18_SEED"], initial_load_model_only=True
    )
    return config


def seed():
    config = _build(_degrees(""))
    config.checkpointer = CheckpointManager.Config(folder=os.environ["MM18_SEED_OUT"])
    config.create_seed_checkpoint = True
    return config
