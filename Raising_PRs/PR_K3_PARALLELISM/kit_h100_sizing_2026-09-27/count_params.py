"""Parameters and static bytes per PP rank of the lb probe at several widths (meta, CPU only).

Usage: PYTHONPATH=<kit>:<tree>:<attn_gym> python count_params.py <pp> <dim> [<dim> ...]
Layout from PPMEM_LAYERS / PPMEM_BLOCK / PPMEM_LPS; stage s lives on rank s % pp (Interleaved1F1B).
Static = 16 bytes per parameter on the rank (fp32 weight, fp32 grad, two AdamW states).
"""
import os
import sys
from collections import defaultdict
from types import SimpleNamespace

import torch

pp = int(sys.argv[1])
dims = [int(x) for x in sys.argv[2:]]
GiB = 2**30
for dim in dims:
    os.environ["PPMEM_DIM"] = str(dim)
    import importlib
    import probe_lb
    importlib.reload(probe_lb)
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
        _get_pipeline_metadata,
    )
    config = probe_lb.lb_probe()
    with torch.device("meta"):
        model = config.model.build()
    per_module = defaultdict(int)
    for name, p in model.named_parameters():
        parts = name.split(".")
        key = ".".join(parts[:2]) if parts[0] == "layers" else parts[0]
        per_module[key] += p.numel()
    config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
    n_stages, n_layers, iw, ow = _get_pipeline_metadata(SimpleNamespace(pp=pp), config.parallelism, config.model)
    split = _generate_llm_fqn_per_model_part(n_stages, n_layers, iw, ow)
    placed = {m for part in split for m in part}
    extra = sorted(k for k in per_module if k not in placed)
    per_rank = defaultdict(int)
    for s, part in enumerate(split):
        for m in part:
            per_rank[s % pp] += per_module.get(m, 0)
    for k in extra:  # the K3 pipeline pins these to the first or the last stage
        r = 0 if "vision" in k else (n_stages - 1) % pp
        per_rank[r] += per_module[k]
    total = sum(per_module.values())
    print(f"dim {dim}: {total/1e6:.0f} M params, {n_stages} stages ({n_layers} layers, in/out weight {iw}/{ow}), "
          f"extra modules {extra}")
    print("   static GiB per rank: " + ", ".join(f"r{r} {16*per_rank[r]/GiB:.2f}" for r in range(pp)))
