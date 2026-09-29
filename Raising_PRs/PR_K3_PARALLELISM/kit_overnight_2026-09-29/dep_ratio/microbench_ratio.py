"""Single-GPU timing of one vision encode against one text pipeline stage (the ratio DEP depends on), measured.

The debug model widened to --dim (tower as wide as the text, 2 layers), bf16, random weights, on one GPU:
- tower: vision_encoder on one image of 224 / 448 / 768 / 1024 px (patch 14, sides 224 / 448 / 756 / 1008):
  forward, and forward + backward (DEP's recompute + backward);
- text: the full model on a text-only micro-batch of --seq tokens, forward and forward + backward, with CUDA events
  around every layer (forward hooks, full backward hooks); a stage is a group of layers of the pp2 x vpp4 split
  (stages 1 to 6 of the eight, the ones without the embedding or the head), reported as their mean.
Prints the forward ratio (encode forward / stage forward) and the step ratio ((encode forward + recompute forward
+ backward) / (stage forward + backward)) per resolution. Warmup 3, measured 10, medians.

Run in the DEP tree: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=<kit>:. python microbench_ratio.py --dim 1024 --seq 2048
"""

import argparse
import os
import statistics

import torch

STAGE_LAYERS = [[2, 3, 4], [5, 6, 7], [8, 9], [10, 11], [12, 13], [14, 15]]
SIDES = {224: 224, 448: 448, 768: 756, 1024: 1008}


def build(dim: int, seq: int, device: str = "cuda"):
    os.environ["DEPW_DIM"] = str(dim)
    import dep_ratio_local

    cfg = dep_ratio_local._widened(enable_sp=False, seq_len=seq)
    with torch.device("meta"):
        model = cfg.build()
    model.to_empty(device=device)
    with torch.no_grad():
        model.init_states()
    return model.to(torch.bfloat16)


def timed(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def tower_times(model, side: int) -> tuple[float, float]:
    per = side // 14
    pix = torch.randn(per * per, 3 * 14 * 14, device="cuda", dtype=torch.bfloat16)
    grid = torch.tensor([[1, per, per]], device="cuda")
    enc = model.vision_encoder

    def fwd():
        with torch.no_grad():
            enc(pix, grid_thw=grid)

    def fwd_bwd():
        out = enc(pix, grid_thw=grid)
        out.float().sum().backward()

    return timed(fwd), timed(fwd_bwd)


def layer_times(model, seq: int) -> tuple[list[float], list[float]]:
    tokens = torch.randint(0, 2000, (seq,), device="cuda")
    positions = torch.arange(seq, device="cuda")
    padding = torch.zeros(seq, dtype=torch.bool, device="cuda")
    masks = model.get_attention_masks(positions, padding_mask=padding, max_context_length=seq)
    layers = list(model.layers.values())
    fwd_ev = [[None, None] for _ in layers]
    bwd_ev = [[None, None] for _ in layers]

    def ev():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    handles = []
    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(lambda m, a, i=i: fwd_ev[i].__setitem__(0, ev())))
        handles.append(layer.register_forward_hook(lambda m, a, o, i=i: fwd_ev[i].__setitem__(1, ev())))
        handles.append(layer.register_full_backward_pre_hook(lambda m, g, i=i: bwd_ev[i].__setitem__(0, ev())))
        handles.append(layer.register_full_backward_hook(lambda m, gi, go, i=i: bwd_ev[i].__setitem__(1, ev())))
    fwd_ms, bwd_ms = [[] for _ in layers], [[] for _ in layers]
    for it in range(13):
        out = model(tokens, positions=positions, attention_masks=masks, padding_mask=padding)
        out.float().mean().backward()
        torch.cuda.synchronize()
        if it >= 3:
            for i in range(len(layers)):
                fwd_ms[i].append(fwd_ev[i][0].elapsed_time(fwd_ev[i][1]))
                if bwd_ev[i][0] is not None and bwd_ev[i][1] is not None:
                    bwd_ms[i].append(bwd_ev[i][0].elapsed_time(bwd_ev[i][1]))
        model.zero_grad(set_to_none=True)
    for h in handles:
        h.remove()
    return [statistics.median(x) for x in fwd_ms], [statistics.median(x) if x else float("nan") for x in bwd_ms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--seq", type=int, default=2048)
    args = ap.parse_args()
    model = build(args.dim, args.seq)
    fwd, bwd = layer_times(model, args.seq)
    stage_f = statistics.mean(sum(fwd[i] for i in s) for s in STAGE_LAYERS)
    stage_fb = statistics.mean(sum(fwd[i] + bwd[i] for i in s) for s in STAGE_LAYERS)
    print(f"dim {args.dim} seq {args.seq}: text stage forward {stage_f:.2f} ms, forward + backward {stage_fb:.2f} ms")
    print("| image px | patches | encode forward ms | encode forward + backward ms | forward ratio | step ratio |")
    print("|---:|---:|---:|---:|---:|---:|")
    for res, side in SIDES.items():
        tf, tfb = tower_times(model, side)
        print(f"| {res} | {(side // 14) ** 2} | {tf:.2f} | {tfb:.2f} | {tf / stage_f:.3f} | {(tf + tfb) / stage_fb:.3f} |")


if __name__ == "__main__":
    main()
