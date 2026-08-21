#!/usr/bin/env python3
"""Which parameters are still plain tensors when FSDP is reached under spmd_types.

    PROBE_ARGS="<train args...>" torchrun --nproc_per_node=2 probe_plain_params.py

spmd_types requires every parameter to already be a DTensor on the full SPMD mesh
before ``fully_shard``, and this model dies there with "Got plain tensor for
parameter 'weight'". That message names one parameter out of thousands, which is
not enough to plan a conversion around -- this dumps the whole inventory, grouped
by owning module type, so the work can be ordered by what is actually missing.

Monkeypatches ``apply_fsdp`` rather than editing the tree: the point is to
measure the tree as it is, and a probe that modifies what it measures is worth
less than no probe.

Prints two tables, both from rank 0 only:
  * per owning-module class: DTensor vs plain counts
  * per parameter-name suffix within the classes that have plain ones
"""

import runpy
import sys
from collections import defaultdict


def _dump(model, when: str) -> None:
    import torch.distributed as dist
    from torch.distributed.tensor import DTensor

    if dist.is_initialized() and dist.get_rank() != 0:
        return

    # Owning module per parameter, so a plain tensor can be attributed to the class
    # that would have to declare it rather than just to a name like "weight".
    owner = {}
    for mod_name, mod in model.named_modules():
        for p_name, _ in mod.named_parameters(recurse=False):
            full = f"{mod_name}.{p_name}" if mod_name else p_name
            owner[full] = type(mod).__name__

    by_class = defaultdict(lambda: [0, 0])  # class -> [dtensor, plain]
    plain_names = defaultdict(lambda: defaultdict(int))  # class -> suffix -> count
    for name, param in model.named_parameters():
        cls = owner.get(name, "?")
        if isinstance(param, DTensor):
            by_class[cls][0] += 1
        else:
            by_class[cls][1] += 1
            plain_names[cls][name.rsplit(".", 1)[-1]] += 1

    # "Declared but nobody applied it" and "never declared" need opposite fixes, and
    # the plain-tensor count alone cannot tell them apart.
    declared = defaultdict(lambda: [0, 0])  # class -> [has config, no config]
    for _, mod in model.named_modules():
        if not any(True for _ in mod.named_parameters(recurse=False)):
            continue
        cfg = getattr(getattr(mod, "config", None), "sharding_config", None)
        cfg = cfg if cfg is not None else getattr(mod, "sharding_config", None)
        declared[type(mod).__name__][0 if cfg is not None else 1] += 1

    total_d = sum(v[0] for v in by_class.values())
    total_p = sum(v[1] for v in by_class.values())
    print(f"\n=== parameters {when} ===", flush=True)
    print(f"  DTensor: {total_d}   plain: {total_p}\n", flush=True)
    print(f"  {'owning module':38s} {'DTensor':>8s} {'plain':>7s}", flush=True)
    for cls, (d, p) in sorted(by_class.items(), key=lambda kv: (-kv[1][1], kv[0])):
        flag = "  <-- blocks spmd_types" if p else ""
        print(f"  {cls:38s} {d:>8d} {p:>7d}{flag}", flush=True)

    print("\n  modules owning parameters, by whether a sharding_config exists:", flush=True)
    print(f"    {'module':36s} {'declared':>9s} {'undeclared':>11s}", flush=True)
    for cls, (yes, no) in sorted(declared.items(), key=lambda kv: (-kv[1][1], kv[0])):
        print(f"    {cls:36s} {yes:>9d} {no:>11d}", flush=True)

    if plain_names:
        print("\n  plain parameter names, by owning module:", flush=True)
        for cls in sorted(plain_names):
            names = ", ".join(
                f"{n} x{c}" for n, c in sorted(plain_names[cls].items())
            )
            print(f"    {cls:36s} {names}", flush=True)
    print("=== end inventory ===\n", flush=True)


def main() -> None:
    import torchtitan.distributed.fsdp as F
    import torchtitan.models.kimi_k3.parallelize as P

    # Patch the FSDP helpers inside the namespaces that call them, and dump ONCE from
    # the first call. Two earlier attempts missed:
    #   * hooking parallelize_kimi_k3 -- the train spec stores the function by value at
    #     import, so rebinding the module attribute changes nothing;
    #   * hooking one apply_fsdp_* helper -- which one trips first depends on the
    #     flavor (multimodal dies in the vision encoder, text in the decoder), so a
    #     single hook measures nothing on the other flavors.
    # The helpers receive a submodule, so walk up to the root via the module the
    # trainer holds; failing that, report what the helper was handed and say so.
    state = {"dumped": set()}

    def wrap(name, owner):
        original = getattr(owner, name, None)
        if original is None:
            return

        def patched(module, *args, **kwargs):
            if name not in state["dumped"]:
                state["dumped"].add(name)
                _dump(module, f"as {name} received it")
            return original(module, *args, **kwargs)

        setattr(owner, name, patched)

    for owner in (P, F):
        for name in ("apply_fsdp_to_vision_encoder", "apply_fsdp_to_decoder"):
            wrap(name, owner)

    # Args come through the environment, not argv: torchrun consumes "--" itself,
    # so a passthrough separator never reaches this script.
    import os
    import shlex

    extra = os.environ.get("PROBE_ARGS")
    if not extra:
        raise SystemExit("set PROBE_ARGS to the torchtitan.train arguments")
    sys.argv = [sys.argv[0]] + shlex.split(extra)
    runpy.run_module("torchtitan.train", run_name="__main__")


if __name__ == "__main__":
    main()
