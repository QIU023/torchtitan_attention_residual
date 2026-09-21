"""LOCAL PROBE HACKS for the PR 4312 round-3 body tables on the H200 (never committed). argv[1] = tree.

On top of probe_apply.py (c4 text-row flavor, seed flavor, naive / 8-stage variants, MB_REVERSE,
NOSYNC_GA, GN_FP32, the KDA capability guard lift, which admits SM90):
* c4 flavors on the PR's own pp2 x vp4 and pp4 x vp4 recipes (their spelled-out splits), naive variants;
* FP32_PROBE=1: bf16 casts stay float32, the experts loop per expert, KDA runs Attention Gym's
  reference implementation (the fused kernels are bf16 only), as on 2026-09-13.
Undo with: git checkout -- <the touched files> in the tree.
"""

import pathlib
import runpy
import sys

tree = pathlib.Path(sys.argv[1])
here = pathlib.Path(__file__).resolve().parent

# 0. the base hacks
runpy.run_path(str(here / "probe_apply.py"), run_name="__main__")

# 1. flavors on the PR's vp4 recipes
p = tree / "torchtitan/models/kimi_k3/config_registry.py"
s = p.read_text()
if "kimi_k3_debugmodel_c4_16stages" not in s:
    s += '''

def _sixteen_stages(config: Trainer.Config) -> Trainer.Config:  # PROBE ONLY (not committed)
    from torchtitan_recipes.tests.b200 import kimi_k3_debugmodel_pp4_vp4

    config.parallelism.pipeline_parallel_module_fqns_per_model_part = (
        kimi_k3_debugmodel_pp4_vp4().parallelism.pipeline_parallel_module_fqns_per_model_part
    )
    return config


def kimi_k3_debugmodel_c4_16stages() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _sixteen_stages(kimi_k3_debugmodel_c4())


def kimi_k3_debugmodel_c4_16stages_naive() -> Trainer.Config:  # PROBE ONLY (not committed)
    return _naive(_sixteen_stages(kimi_k3_debugmodel_c4()))
'''
    p.write_text(s)

# 2. FP32_PROBE
p = tree / "torchtitan/trainer.py"
s = p.read_text()
if "FP32_PROBE" not in s:
    anchor = "import torch\nimport tyro\n"
    assert s.count(anchor) == 1
    s = s.replace(
        anchor,
        '''import torch

if __import__("os").environ.get("FP32_PROBE") == "1":  # LOCAL PROBE HACK (not committed): bf16 casts stay float32
    torch.Tensor.bfloat16 = lambda self, *a, **k: self
    _fp32_probe_to = torch.Tensor.to

    def _fp32_probe_to_fn(self, *a, **k):
        a = tuple(torch.float32 if x is torch.bfloat16 else x for x in a)
        if k.get("dtype") is torch.bfloat16:
            k["dtype"] = torch.float32
        return _fp32_probe_to(self, *a, **k)

    torch.Tensor.to = _fp32_probe_to_fn
import tyro
''',
        1,
    )
    p.write_text(s)

p = tree / "torchtitan/models/common/moe.py"
s = p.read_text()
if "FP32_PROBE" not in s:
    anchor = "        return torch._grouped_mm(A, weight_EOI.bfloat16().transpose(-2, -1), offs=offs)\n"
    assert s.count(anchor) == 1
    s = s.replace(
        anchor,
        '''        if __import__("os").environ.get("FP32_PROBE") == "1":  # LOCAL PROBE HACK (not committed): per-expert loop
            W, pieces, start = weight_EOI.transpose(-2, -1), [], 0
            for e, end in enumerate(offs.tolist()):
                pieces.append(A[start:end] @ W[e].to(A.dtype))
                start = end
            pieces.append(A.new_zeros(A.shape[0] - start, W.shape[-1]))
            return torch.cat(pieces)
'''
        + anchor,
        1,
    )
    p.write_text(s)

p = tree / "torchtitan/models/kimi_k3/kda.py"
s = p.read_text()
if "FP32_PROBE" not in s:
    a1 = '            impl="fused",\n'
    assert s.count(a1) == 1
    s = s.replace(a1, '            impl="reference" if __import__("os").environ.get("FP32_PROBE") == "1" else "fused",  # LOCAL PROBE HACK (not committed)\n', 1)
    a2 = "            cu_seqlens=cu_seqlens,\n        )\n        return output_1THV\n"
    assert s.count(a2) == 1, s.count(a2)
    s = s.replace(
        a2,
        '            cu_seqlens=cu_seqlens,\n            **({"impl": "reference"} if __import__("os").environ.get("FP32_PROBE") == "1" else {}),  # LOCAL PROBE HACK (not committed)\n        )\n        return output_1THV\n',
        1,
    )
    p.write_text(s)

# 4. the reference logs its step loss the way the pipeline's last stage does (torch.sum(torch.stack(losses)))
p = tree / "torchtitan/trainer.py"
s = p.read_text()
if "_PROBE_LOSSES" not in s:
    old = """            if should_log:
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)
"""
    new = """            if should_log and os.environ.get("NOSYNC_GA") == "1":  # LOCAL PROBE HACK (not committed): the reference logs its step loss the way the pipeline's last stage does, torch.sum(torch.stack(losses)), not a running add
                _probe_losses = globals().setdefault("_PROBE_LOSSES", [])
                _probe_losses.append(detached_loss.clone())
                if fwd_bwd_index == len(microbatch_groups) - 1:
                    accumulated_loss = torch.sum(torch.stack(_probe_losses)).to(detached_loss.device)
                    _probe_losses.clear()
            elif should_log:
                if accumulated_loss is None:
                    # Take ownership before the next replay overwrites the
                    # graph-owned output. Later losses accumulate in place.
                    accumulated_loss = detached_loss.clone()
                else:
                    accumulated_loss.add_(detached_loss)
"""
    assert s.count(old) == 1
    p.write_text(s.replace(old, new, 1))
print("H200 probe hacks applied to", tree)
