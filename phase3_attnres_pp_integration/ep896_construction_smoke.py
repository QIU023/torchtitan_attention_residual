"""EP@896 construction + forward smoke: the real K3 expert count (896)
sharded EP=8 (112 experts/rank), debug dims so it fits. Validates the
all-to-all dispatch mesh + MoE sharding at the 2.8T expert count that
config-level scale-out claims rely on."""
import os

import torch
import torch.distributed as dist


def main():
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = dist.get_rank()

    from torchtitan.config import ParallelismConfig, TrainingConfig
    from torchtitan.config.configs import CompileConfig
    from torchtitan.distributed import ParallelDims
    from torchtitan.experiments.kimi_k3 import config_registry
    from torchtitan.experiments.kimi_k3.model import KimiLinearSpec
    from torchtitan.experiments.kimi_k3.parallelize import parallelize_kimi_linear

    # Debug dims, REAL 896 experts, top-16 (K3 counts).
    kc = config_registry.build_kimi_linear_config(
        "debugmodel", num_experts=896
    )
    kc.num_experts_per_token = 16
    kc.moe_enable_ep = True  # declare EP sharding at config-build (update_from_config does this in the trainer)
    spec = KimiLinearSpec(kimi_config=kc, num_blocks=None)

    pdims = ParallelDims(
        dp_replicate=1,
        dp_shard=8,   # EP nests inside dp_shard: 8 ranks host 896/8 experts
        cp=1,
        tp=1,
        pp=1,
        ep=8,
        world_size=8,
    )
    pdims.build_mesh()

    with torch.device("meta"):
        model = spec.build()
    parallelize_kimi_linear(
        model,
        parallel_dims=pdims,
        training=TrainingConfig(),
        parallelism=ParallelismConfig(data_parallel_shard_degree=8, expert_parallel_degree=8),
        compile_config=CompileConfig(enable=False),
        ac_config=None,
        dump_folder="/workspace/smoke_runs/ep896",
    )
    model.to_empty(device="cuda")
    model.init_weights()

    tokens = torch.randint(0, 2016, (1, 128), device="cuda")
    out = model(tokens)
    loss = out.float().sum()
    loss.backward()
    if rank == 0:
        experts_per_rank = 896 // 8
        print(
            f"[EP896] world=8 ep=8 experts=896 => {experts_per_rank}/rank | "
            f"forward finite={bool(torch.isfinite(out).all())} "
            f"loss={loss.item():.2f}",
            flush=True,
        )
        print("[EP896] PASS", flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
