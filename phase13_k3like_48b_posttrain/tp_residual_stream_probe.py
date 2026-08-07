"""Where does the TP residual stream first diverge across ranks?

The LoRA investigation ended at a premise failure it could not localise: at
layer 1's ``q_a_proj`` the module input ``x`` is LABELLED ``Replicate`` yet
``inter = lora_a(x)`` differs across TP ranks by 57% of its magnitude, while at
layer 0 both are identical. Since ``lora_a`` is Replicate, only ``x`` can be
responsible -- so something between layer 0's input and layer 1's input turns a
replicated activation into a rank-dependent one.

This probe does not test a hypothesis. It records, in EXECUTION ORDER, every
module output in layers 0 and 1, its placement, and its cross-rank delta, and
lets the first unexpected divergence name the site. Guessing between o_proj,
down_proj and the AttnRes bridges is what this replaces.

Reading the output requires the classification, because divergence is CORRECT in
most of these places:

* ``Shard`` / ``Partial`` DTensors are supposed to differ -- each rank holds its
  own piece or its own addend.
* Plain tensors produced by a ``use_local_output=True`` style are the local
  result of a collective and should agree; plain tensors produced INSIDE the
  sharded region (SDPA output, the per-head gate) are per-rank pieces and should
  not. The table prints the value so the classification is visible rather than
  assumed, and marks with ``EXPECT-DIFF`` only the names known to be inside the
  sharded region.
* ``Replicate`` DTensors and the residual stream must agree. A divergence there
  is the defect.

Run (from the titan checkout, tp2, one step):

    P=<logbook>/phase13_k3like_48b_posttrain
    PYTHONPATH=$PWD:$P torchrun --nproc_per_node=2 --master_port=49611 \
      $P/tp_residual_stream_probe.py --module kimi_k3 \
      --config kimi_k3_mini_diag_4l_mla_lora --debug.seed 42 \
      --debug.deterministic --training.steps 1 --training.global-batch-size 4 \
      --parallelism.data_parallel_shard_degree 1 \
      --parallelism.tensor_parallel_degree 2 \
      --checkpoint.interval 100000 --dump-folder /workspace/out_resid_probe

Swap to ``kimi_k3_mini_diag_4l_mla`` for the LoRA-off control: if the same site
diverges there, LoRA is not involved at all and every TP leg of the published
matrices is affected.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

# Modules whose output legitimately differs per rank because it is produced
# inside the sharded region rather than at a collective boundary.
EXPECT_DIFF = (
    "inner_attention",  # SDPA over this rank's heads
    "attn_gate_proj",  # Colwise, use_local_output=True -> this rank's gate shard
    "q_b_proj",
    "kv_b_proj",  # Colwise, head-sharded
    "gate_proj",
    "up_proj",  # Colwise, feature-sharded
)

TRACE_PREFIXES = ("layers.0", "layers.1")

# Modules whose INPUT is also recorded, plus a one-line dump of the module's
# class, children and weight placement -- so "is this even LoRA-wrapped, and did
# it inherit the divergence or produce it" is answered from the same run.
TRACE_INPUTS = ("o_proj", "down_proj")

_ORDER: list[tuple[int, str, str, float, float, bool]] = []
_SEQ = [0]


def _describe(t: torch.Tensor) -> str:
    if isinstance(t, DTensor):
        return "+".join(str(p)[0] for p in t.placements)  # R / S / P
    return "plain"


def _compare(t: torch.Tensor) -> tuple[float, float]:
    """Max cross-rank delta and rank 0's magnitude, on the LOCAL values."""
    local = (t.to_local() if isinstance(t, DTensor) else t).detach().float()
    local = local.contiguous()
    shapes = [None] * dist.get_world_size()
    dist.all_gather_object(shapes, tuple(local.shape))
    if len(set(shapes)) != 1:
        # Different local shapes: a sharded value on an uneven axis. Not
        # comparable, and saying so is the point -- silence must not stand in
        # for agreement.
        return (float("nan"), float("nan"))
    buf = [torch.empty_like(local) for _ in range(dist.get_world_size())]
    dist.all_gather(buf, local)
    delta = max((o - buf[0]).abs().max().item() for o in buf[1:])
    return (delta, buf[0].abs().max().item())


def _record(name: str, t: torch.Tensor) -> None:
    if not isinstance(t, torch.Tensor):
        return
    if t.numel() == 0:
        return
    delta, mag = _compare(t)
    expected = any(m in name for m in EXPECT_DIFF)
    _SEQ[0] += 1
    _ORDER.append((_SEQ[0], name, _describe(t), delta, mag, expected))


def _first_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, (tuple, list)):
        for item in x:
            found = _first_tensor(item)
            if found is not None:
                return found
    return None


def _install(part: torch.nn.Module) -> int:
    n = 0
    for name, mod in part.named_modules():
        short = name.split("model.")[-1]
        if not short.startswith(TRACE_PREFIXES):
            continue
        if len(list(mod.children())) and not short.count(".") <= 1:
            # Keep containers only at the layer level; deeper containers add
            # duplicate rows for the same value.
            pass

        def hook(_m, inp, out, _name=short):
            if any(m in _name for m in TRACE_INPUTS):
                t = _first_tensor(inp)
                if t is not None:
                    _record(f"{_name} IN", t)
                    if dist.get_rank() == 0:
                        w = getattr(_m, "weight", None)
                        print(
                            f"PROBE-MOD {_name} cls={type(_m).__name__} "
                            f"children={[c for c, _ in _m.named_children()]} "
                            f"w={_describe(w) if w is not None else 'none'}",
                            flush=True,
                        )
            t = _first_tensor(out)
            if t is not None:
                _record(f"{_name} OUT", t)
            return None

        mod.register_forward_hook(hook)
        n += 1
    return n


def main() -> None:
    import torchtitan.train as T

    _init = T.Trainer.__init__

    def patched(self, *a, **k):
        _init(self, *a, **k)
        total = sum(_install(p) for p in self.model_parts)
        if dist.get_rank() == 0:
            print(f"PROBE installed {total} hooks on {TRACE_PREFIXES}", flush=True)
        step = self.train_step

        def wrapped(*a, **k):
            _ORDER.clear()
            _SEQ[0] = 0
            r = step(*a, **k)
            if dist.get_rank() == 0:
                print(
                    f"\n{'seq':>4} {'module':52s} {'place':8s} "
                    f"{'delta':>11s} {'mag':>11s}  verdict",
                    flush=True,
                )
                first_bad = None
                for seq, name, place, delta, mag, expected in _ORDER:
                    if delta != delta:  # nan
                        verdict = "NOT-COMPARABLE"
                    elif delta == 0.0:
                        verdict = "identical"
                    elif expected:
                        verdict = "differs (EXPECT-DIFF)"
                    else:
                        verdict = "DIFFERS  <-- unexpected"
                        if first_bad is None:
                            first_bad = (seq, name, place, delta, mag)
                    print(
                        f"{seq:>4} {name:52s} {place:8s} "
                        f"{delta:11.3e} {mag:11.3e}  {verdict}",
                        flush=True,
                    )
                if first_bad:
                    print(
                        f"\nPROBE first unexpected divergence: seq={first_bad[0]} "
                        f"{first_bad[1]} placement={first_bad[2]} "
                        f"delta={first_bad[3]:.3e} mag={first_bad[4]:.3e}",
                        flush=True,
                    )
                else:
                    print(
                        "\nPROBE no unexpected divergence in the traced range",
                        flush=True,
                    )
            return r

        self.train_step = wrapped

    T.Trainer.__init__ = patched

    from torchtitan.train import main as titan_main

    titan_main()


if __name__ == "__main__":
    os.environ.setdefault("TORCH_NCCL_AVOID_RECORD_STREAMS", "1")
    main()
