# Item 6 re-run: LoRA and CP do not compose on the current tree

The plan closed item 6, LoRA plus context parallel, on 2026-09-18, with the runtime half taken on `/tmp/wt_verl_0915` at `549c1e21` and the standing caveat over all of items 6 to 9: "A pass on an older tree says the capability worked then, not that it works today." Re-running it on the current pair, verl `b5a79e15` and the integration tree `cf7418637`, is what that caveat asks for, and it does not pass.

    VERL_TORCHTITAN_FLAVOR=kimi_k3_rl_lora CP_SIZE=2 FSDP_SIZE=2 NUM_GPUS=8

    ValueError: UlyssesCPFlexInnerAttention.Config must inherit
                FrozenFlexInnerAttention.Config.

The raise is torchtitan's own, at `torchtitan/config/transform/base.py:45`, reached from `config/transform/context_parallel.py:55` through `apply_transforms`. `convert_config_type(existing, replacement)` requires `replacement.Config` to be a subclass of the **existing** config's type, and its docstring says why: "Requiring inheritance preserves wrappers added by earlier transforms."

So the name in the message is not a class the engine asks for, it is the class the model already has. `FrozenFlexInnerAttention` does not exist anywhere in the tree as a source definition; it is what the LoRA transform leaves behind when it freezes the base module. The CP transform then tries to convert that wrapper into `UlyssesCPFlexInnerAttention`, whose `Config` inherits `UlyssesCPInnerAttention.Config` and `FlexInnerAttention.Config` and not the frozen wrapper's, and torchtitan refuses.

That is exactly item 6's subject seen from the other side. The item is about transform ordering, the flavor's LoRA converter running before the engine's context-parallel transform, and the static half of it was closed by establishing that `apply_transforms` sorts on `run_after` rather than on append order, so LoRA does run first. The runtime half now says what that ordering costs: with LoRA applied first, the context-parallel transform cannot convert the frozen config, and the combination does not start.

Two things this does not settle. Whether the fix belongs in torchtitan, by making the frozen wrapper's config participate in the inheritance the converter wants, or in the engine, by transforming for CP before LoRA freezes anything, is a design question for whoever owns the ordering. And whether the old tree passed because the frozen wrapper did not exist then, or because the transform order differed, is not established here; the only claim is that the pair as it stands today refuses.

## The other half of the same run, and a false green

Item 8, QLoRA with the fused `w13` projection, was launched alongside and failed for a reason of mine rather than the tree's: `AssertionError: Invalid parallel dims: dp_replicate(1) * dp_shard(2) * cp(1) * tp(1) * pp(1) != WORLD_SIZE(8)`. `FSDP_SIZE=2` with eight GPUs does not multiply to the world size. Re-run with `FSDP_SIZE=8`.

Worth recording separately: the chain reported `rc=0` for both cells and both had failed. The wrapper captured the exit status of the launcher, which returns zero whether or not the inner job reached a step, so a pair of one-minute cells with the GPUs never touched read as passes. The tell that caught it was wall-clock and `nvidia-smi`, not the status. A marker has to gate on the thing: for a training cell that means a step count, not an exit code.

## Before that, the run did not start at all

The first launch sat thirty minutes after "worker group kwargs" with every GPU at zero percent and no error. `matrix_scripts/verl_grpo_int0916_nd.sh:47` passes `ray_kwargs.ray_init.num_cpus=${RAY_CPUS:-24}` on a 64-core box, and `ray.available_resources()` showed all eight GPUs still available, which is the signature of a placement group that never formed. `RAY_CPUS=48` gets past it.
