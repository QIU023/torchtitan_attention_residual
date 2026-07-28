"""veRL actor smoke: can veRL's torchtitan engine train our K3 model?

veRL 0.8.0 ships a torchtitan engine (verl/workers/engine/torchtitan/), so our
model can be the actor directly -- no HF wrapper, no Megatron. This drives the
real path: HFModelConfig -> flavor derivation -> TorchTitanEngine -> a forward
and backward.

Three shims, all of which belong upstream in veRL rather than in our tree:

* its model_type map has no "kimi_k3" entry;
* its engine hardcodes the ``torchtitan.models.`` prefix, and ours lives under
  ``experiments`` (CLAUDE.md: stay in experiments until a maintainer asks);
* ``peft`` is imported eagerly by verl's package init and needs
  ``BloomPreTrainedModel``, gone in transformers 5.x. The titan engine itself
  does not use peft -- that is veRL's LoRA support -- so it is stubbed.

RESULT (2026-07-28, veRL 0.8.0): gets as far as engine construction and stops on
a genuine API skew, not a shim-able one.

    [VERL] hf_config: type=kimi_k3 hidden=512 layers=21 vocab=163840
    [VERL] derived name=kimi_k3 flavor=kimi_linear_k3mini_block_attn_res
    TypeError: OptimizersContainer.Config.__init__() got an unexpected
               keyword argument 'name'

So flavor resolution works end to end through real veRL machinery, off a real HF
config. What blocks the train step is that veRL's titan engine builds the
optimizer against an OLDER torchtitan API::

    verl/workers/engine/torchtitan/transformer_impl.py:113
        OptimizersContainer.Config(name=..., lr=..., eps=..., beta1=...,
                                   beta2=..., weight_decay=...)

while the torchtitan revision this fork tracks takes
``OptimizersContainer.Config(param_groups=..., implementation=...)`` -- the
param-group API that ``default_adamw`` builds. The two cannot both be satisfied,
and we cannot pin the older torchtitan because the K3 work depends on the newer
one (DataParallelMeshDims, the shape-suffixed GroupedExperts params, the
declarative init map).

That makes it a veRL-side fix: either its engine adapts to the param-group API,
or torchtitan keeps a compat path. Deliberately NOT worked around here -- forcing
it would mean constructing veRL's optimizer ourselves and bypassing the engine,
which would smoke our own code rather than the integration.

Launch: PYTHONPATH=<titan> torchrun --nproc_per_node=2 verl_actor_smoke.py
"""

from __future__ import annotations

import os
import sys
import types

import torch


def install_shims() -> None:
    for name in ("peft", "peft.utils", "peft.utils.constants"):
        m = types.ModuleType(name)
        m.__path__ = []
        sys.modules.setdefault(name, m)
    for attr in ("PeftModel", "LoraConfig", "TaskType", "PeftConfig"):
        setattr(sys.modules["peft"], attr, object)
    sys.modules["peft"].get_peft_model = lambda *a, **k: None

    import torchtitan.experiments.kimi_k3 as k3

    sys.modules.setdefault("torchtitan.models.kimi_k3", k3)

    from verl.workers.engine.torchtitan import utils as vu

    vu._HF_MODEL_TYPE_TO_TORCHTITAN_NAME["kimi_k3"] = "kimi_k3"


def main() -> None:
    install_shims()
    import torch.distributed as dist

    from verl.workers.config.model import HFModelConfig

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    model_config = HFModelConfig(
        path="/workspace/k3mini_hf",
        load_tokenizer=False,
        trust_remote_code=True,
    )
    if rank == 0:
        hc = model_config.hf_config
        print(
            f"[VERL] hf_config: type={hc.model_type} hidden={hc.hidden_size} "
            f"layers={hc.num_hidden_layers} vocab={hc.vocab_size}",
            flush=True,
        )

    from verl.workers.engine.torchtitan.utils import (
        derive_torchtitan_name_and_flavor,
    )

    name, flavor = derive_torchtitan_name_and_flavor(model_config.hf_config)
    if rank == 0:
        print(f"[VERL] derived name={name} flavor={flavor}", flush=True)

    from verl.workers.config.engine import TorchtitanEngineConfig
    from verl.workers.config.optimizer import TorchtitanOptimizerConfig
    from verl.trainer.config.config import CheckpointConfig
    from verl.workers.engine.torchtitan.transformer_impl import (
        TorchTitanEngineWithLMHead,
    )

    engine_config = TorchtitanEngineConfig()
    optimizer_config = TorchtitanOptimizerConfig(lr=1e-4)
    checkpoint_config = CheckpointConfig()

    engine = TorchTitanEngineWithLMHead(
        model_config=model_config,
        engine_config=engine_config,
        optimizer_config=optimizer_config,
        checkpoint_config=checkpoint_config,
    )
    if rank == 0:
        print(f"[VERL] engine constructed: {type(engine).__name__}", flush=True)
    engine.initialize()
    if rank == 0:
        n = sum(
            p.numel() for m in engine.module for p in m.parameters()
        )
        print(f"[VERL] initialized, {n/1e6:.1f}M params on this rank", flush=True)
        print("[VERL] PASS (engine built and initialized)", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
