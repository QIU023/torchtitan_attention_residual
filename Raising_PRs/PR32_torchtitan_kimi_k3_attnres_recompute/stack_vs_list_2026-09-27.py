"""CPU check: what each layer's call-site remat.checkpoint keeps alive, stack carrier vs list carrier.

Model: L layers, a block opens every B layers; each layer applies the attention residual twice
(attention_res, ffn_res), each wrapped in torch_remat.checkpoint at the call site, as #4656 does.
After forward, only the loss is referenced; count distinct live block storages kept by autograd.
"""
import gc
import weakref

import torch
import torch.nn as nn
import torch_remat as remat

T, D, L, B = 64, 32, 24, 4  # 6 blocks


class Norm(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(D))
        self.eps = 1e-6


def agg_stack(partial_TD, block_residual_TND, projection, norm):  # main e033f7517
    values_TND = (
        block_residual_TND
        if partial_TD is None
        else torch.cat((block_residual_TND, partial_TD.unsqueeze(1)), dim=1)
    )
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    return torch.matmul(probs_T1N, values_float).squeeze(1).to(values_TND.dtype)


def agg_list(partial_TD, blocks_TD, projection, norm):  # PR A 7ae870508
    values_TND = torch.stack(
        blocks_TD if partial_TD is None else [*blocks_TD, partial_TD], dim=1
    )
    values_float = values_TND.float()
    variance = values_float.pow(2).mean(dim=-1, keepdim=True)
    keys_TND = values_float * torch.rsqrt(variance + norm.eps)
    score_weight_D = norm.weight.float() * projection.weight.squeeze(0).float()
    scores_TN = (keys_TND * score_weight_D).sum(dim=-1)
    probs_T1N = torch.softmax(scores_TN, dim=-1).unsqueeze(1)
    return torch.matmul(probs_T1N, values_float).squeeze(1).to(values_TND.dtype)


def run(carrier: str, checkpoint: bool):
    torch.manual_seed(0)
    projs = [nn.Linear(D, 1, bias=False, dtype=torch.bfloat16) for _ in range(2 * L)]
    norms = [Norm() for _ in range(2 * L)]
    mix = [nn.Linear(D, D, bias=False, dtype=torch.bfloat16) for _ in range(L)]
    x = torch.randn(T, D, dtype=torch.bfloat16, requires_grad=True)
    agg = agg_stack if carrier == "stack" else agg_list
    refs = []  # weakrefs to every tensor that carries block data
    carrier_obj = x.unsqueeze(1)[:, :0] if carrier == "stack" else []
    prefix = None
    h = x
    for layer in range(L):
        opens = layer % B == 0
        if opens:
            if carrier == "stack":
                carrier_obj = torch.cat((carrier_obj, h.unsqueeze(1)), dim=1)
                refs.append(weakref.ref(carrier_obj))
            else:
                carrier_obj = [*carrier_obj, h]
                refs.append(weakref.ref(h))
            partial = None
        else:
            partial = h
        call = (
            (lambda *a, name=f"l{layer}a": remat.checkpoint(region_name=name)(agg)(*a))
            if checkpoint
            else agg
        )
        h = call(partial, carrier_obj, projs[2 * layer], norms[2 * layer])
        h = torch.tanh(mix[layer](h))
        call2 = (
            (lambda *a, name=f"l{layer}f": remat.checkpoint(region_name=name)(agg)(*a))
            if checkpoint
            else agg
        )
        h = h + call2(h, carrier_obj, projs[2 * layer + 1], norms[2 * layer + 1])
    loss = h.float().square().mean()
    del carrier_obj, h, partial, call, call2
    gc.collect()
    alive = [r() for r in refs if r() is not None]
    storages = {}
    for t in alive:
        s = t.untyped_storage()
        storages[s.data_ptr()] = s.nbytes()
    unit = T * D * 2
    live_units = sum(storages.values()) / unit
    loss.backward()
    return len(alive), live_units


for carrier in ("stack", "list"):
    for ckpt in (True, False):
        n, units = run(carrier, ckpt)
        print(f"{carrier:5s} checkpoint={ckpt!s:5s}: {n} block carriers alive after forward, "
              f"{units:.0f} [T, D] units of block storage held (blocks: {L // B}, longest stack {L // B} units)")


def run_block_ac(carrier: str):
    """Whole-layer checkpoint (full AC's shape): the layer body takes (h, carrier) as inputs."""
    torch.manual_seed(0)
    projs = [nn.Linear(D, 1, bias=False, dtype=torch.bfloat16) for _ in range(2 * L)]
    norms = [Norm() for _ in range(2 * L)]
    mix = [nn.Linear(D, D, bias=False, dtype=torch.bfloat16) for _ in range(L)]
    x = torch.randn(T, D, dtype=torch.bfloat16, requires_grad=True)
    agg = agg_stack if carrier == "stack" else agg_list
    refs = []

    def body(layer, h, carrier_obj):
        opens = layer % B == 0
        if opens:
            if carrier == "stack":
                carrier_obj = torch.cat((carrier_obj, h.unsqueeze(1)), dim=1)
            else:
                carrier_obj = [*carrier_obj, h]
        partial = None if opens else h
        out = agg(partial, carrier_obj, projs[2 * layer], norms[2 * layer])
        out = torch.tanh(mix[layer](out))
        out = out + agg(out, carrier_obj, projs[2 * layer + 1], norms[2 * layer + 1])
        return out, carrier_obj

    carrier_obj = x.unsqueeze(1)[:, :0] if carrier == "stack" else []
    h = x
    for layer in range(L):
        # the new carrier is built outside the region here, as the model's forward returns it
        if layer % B == 0:
            if carrier == "stack":
                carrier_obj = torch.cat((carrier_obj, h.unsqueeze(1)), dim=1)
                refs.append(weakref.ref(carrier_obj))
            else:
                carrier_obj = [*carrier_obj, h]
                refs.append(weakref.ref(h))
        out = remat.checkpoint(region_name=f"layers.{layer}")(
            lambda h_, c_, layer=layer: body_no_open(layer, h_, c_)
        )(h, carrier_obj)
        h = out

    loss = h.float().square().mean()
    del carrier_obj, h, out
    gc.collect()
    alive = [r() for r in refs if r() is not None]
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in alive}
    units = sum(storages.values()) / (T * D * 2)
    loss.backward()
    return len(alive), units


def body_no_open(layer, h, carrier_obj):
    agg = agg_stack if not isinstance(carrier_obj, list) else agg_list
    partial = None if layer % B == 0 else h
    idx = 2 * layer
    torch.manual_seed(layer)
    proj = nn.Linear(D, 1, bias=False, dtype=torch.bfloat16)
    norm = Norm()
    out = agg(partial, carrier_obj, proj, norm)
    return torch.tanh(out)


for carrier in ("stack", "list"):
    n, units = run_block_ac(carrier)
    print(f"{carrier:5s} whole-layer checkpoint: {n} block carriers alive, {units:.0f} [T, D] units held")
