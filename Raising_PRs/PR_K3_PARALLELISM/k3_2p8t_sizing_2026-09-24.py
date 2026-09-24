"""Feasible 5D layouts for 2.8T Kimi K3 pre-training on H100 and GB300: what fits, and what it costs.

Run from a checkout of pp_review4 (for the layout tables) with torch nightly:
    PYTHONPATH=<pp_review4>:. python k3_2p8t_sizing_2026-09-24.py

Fixed by the model (in-tree Kimi-K3 flavor) and the report (section 5.2):
  - 93 layers + 1 MTP layer (report 3.3), dim 7168, 96 heads in MLA and KDA, 896 experts top-16,
    latent 3584, expert hidden 3072, 2 shared experts; 8K tokens per sequence in pre-training.
  - Parallelism: PP with VP, EP, ZeRO-1 DP, Pipeline ZeRO-2 (grads on CPU, a double grad buffer
    on GPU), CP. No TP is listed.
Not given anywhere: the cluster size N, the global batch, the degrees. Everything below is a
feasibility range, not a recovered configuration.

Rank accounting (titan's ParallelDims agrees): N = PP x TP x CP x DP. EP takes no ranks of its
own: it borrows from DP x CP x TP, and the expert optimizer shards over N / (PP x EP).
"""

import importlib.util
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location(
    "timeline", os.path.join(HERE, "pp_memory_timeline_2026-09-24.py")
)
timeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(timeline)

GiB = 2**30
DIM, LAYERS, HEADS, EXPERTS, TOPK, LATENT, EXPERT_HIDDEN = 7168, 93, 96, 896, 16, 3584, 3072
SEQ = 8192
EXPERT_PARAMS = 3 * LATENT * EXPERT_HIDDEN  # one routed expert, 33.0 M
MLA_LAYERS = timeline.MLA_LAYERS


def nonexpert_b(layer):
    return timeline.nonexpert(layer)


EXPERTS_PER_LAYER_B = EXPERTS * EXPERT_PARAMS / 1e9  # 29.6 B
MTP_B = EXPERTS_PER_LAYER_B + 0.634  # one backbone block (report 3.3)
EMBED_B, HEAD_B, VISION_B = timeline.EMBED, timeline.HEAD, timeline.VISION

# Per layer and micro-batch, what backward keeps under the report's recipe (FP8 block-wise
# activations, element-wise ops recomputed, dispatch recomputed, AttnRes checkpointed):
# attention inputs and q/k/v/g/beta/out in FP8, MoE input, shared-expert gate/up, the latent,
# the routed gate/up in FP8 (top-16 x 2 x 3072), and the layer input in bf16. Bytes per token.
ACT_BYTES_PER_TOKEN = (
    DIM + 5 * HEADS * 128  # attention: normed input + q, k, v, decay, output (FP8)
    + DIM + 3 * 2 * EXPERT_HIDDEN  # MoE input + shared experts gate/up/act (FP8)
    + 2 * LATENT  # latent in and out (FP8)
    + TOPK * 2 * EXPERT_HIDDEN  # routed gate/up (FP8)
    + 2 * DIM  # layer input, bf16
)

HW = {
    # HBM usable, NVLink domain, NVLink GB/s per direction, IB GB/s per GPU, host link GB/s,
    # effective BF16 TFLOPS on MoE GEMMs (about half of dense peak)
    "H100": dict(hbm=80e9 / GiB, domain=8, nvlink=450, ib=50, host=55, tflops=450),
    "GB300": dict(hbm=288e9 / GiB, domain=72, nvlink=900, ib=50, host=450, tflops=1100),
}
RESERVED_GIB = 4.0  # CUDA context, NCCL, allocator slack
GRAD_DOUBLE_BUFFER_GB = 2.0  # bucket-level double buffer (the report gives no size)


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def rank_layers(pp, vp):
    _, l2s = timeline.layout_split(pp, vp)
    S = pp * vp
    return {r: [layer for layer, st in l2s.items() if st % pp == r] for r in range(pp)}, S


def static_gib(pp, vp, ep, tp, n):
    """Heaviest rank: bf16 params, ZeRO-1 fp32 master + momentum shards, the double grad buffer,
    MoonEP's reserved E/EP redundant-expert slots (weights bf16 + grad staging fp32) and its
    fixed S x K dispatch/combine buffer."""
    layers, S = rank_layers(pp, vp)
    worst = None
    for r in range(pp):
        moe = [layer for layer in layers[r] if layer]
        experts = len(moe) * EXPERTS_PER_LAYER_B / ep
        dense = sum(nonexpert_b(layer) for layer in layers[r]) / tp
        if r == 0:
            dense += (EMBED_B + VISION_B) / tp
        if r == (S - 1) % pp:
            dense += HEAD_B / tp + 0.634 / tp
            experts += EXPERTS_PER_LAYER_B / ep
        params = 2 * (experts + dense)
        optim = 8 * experts / (n / (pp * ep)) + 8 * dense / (n / (pp * tp))
        grad_buf = GRAD_DOUBLE_BUFFER_GB
        redundant = (EXPERTS / ep) * EXPERT_PARAMS * (2 + 4) / 1e9
        dispatch = SEQ / tp * TOPK * LATENT * 3 / 1e9
        total = (params + optim + grad_buf + redundant + dispatch) * 1e9 / GiB + RESERVED_GIB
        if worst is None or total > worst[0]:
            worst = (total, r, params * 1e9 / GiB, experts, dense)
    return worst


def act_gib_per_layer(tp, cp):
    return ACT_BYTES_PER_TOKEN * SEQ / (tp * cp) / GiB


def in_flight(pp, vp, m):
    acts, _ = timeline.activation_timeline(pp, vp, m)
    return max(max(v) for v in acts.values())


def attnres_lower_gib(pp, vp, m, tp):
    per, dyn = timeline.block_timeline(pp, vp, m, "lower")
    hid = timeline.hidden_buffers(pp, vp, m)
    unit = SEQ / tp * DIM * 2 / GiB
    return max(unit * (hid[r] + per[r] + max(dyn[r])) for r in range(pp))


def ep_ib_ms(ep, tp, hw):
    """Forward all-to-all bytes per layer per micro-batch that leave the NVLink domain, over IB."""
    tokens = SEQ / tp
    fwd = tokens * TOPK * LATENT * (1 + 2)  # FP8 dispatch + bf16 combine
    in_domain = min(ep, hw["domain"]) / ep
    return fwd * (1 - in_domain) / (hw["ib"] * 1e9) * 1e3


def layer_fwd_ms(tp, hw):
    active = 1.16e9  # per KDA layer: attention 0.443 B + shared 0.132 B + latent 0.051 B + 16 experts 0.528 B
    return 2 * active * SEQ / tp / (hw["tflops"] * 1e12) * 1e3


def main():
    print(f"act per layer per 8K micro-batch (report recipe, estimate): {act_gib_per_layer(1, 1):.2f} GiB")
    print(f"params: experts {LAYERS - 1} x {EXPERTS_PER_LAYER_B:.1f} B + MTP {MTP_B:.1f} B")
    for name, hw in HW.items():
        print(f"\n=== {name}: HBM {hw['hbm']:.1f} GiB, NVLink domain {hw['domain']}, IB {hw['ib']} GB/s per GPU")
        tps = [t for t in divisors(HEADS) if t <= min(8, hw["domain"])]
        print(f"TP candidates (divide 96 heads, inside one NVLink domain, at most 8): {tps}")
        print("| PP x VP | EP | TP | stages | layers/stage | static GiB (rank) | layer x mb in flight (M) "
              "| acts no offload GiB | AttnRes+hidden (lower bound) GiB | headroom for acts GiB "
              "| acts that must leave HBM | EP IB ms / layer fwd ms |")
        print("|---|---:|---:|---:|---:|---|---|---:|---:|---:|---:|---|")
        n = 4096
        for pp in (2, 4, 8, 16, 32):
            for vp in (1, 2, 3, 4, 6, 8):
                S = pp * vp
                if S > LAYERS // 2 or S < 8:
                    continue
                m = max(2 * pp, 16) if pp <= 8 else 2 * pp
                m = (m // pp) * pp
                for tp in tps:
                    if tp not in (1, 2, 8):
                        continue
                    for ep in (8, 16, 32, 64):
                        if ep < tp or EXPERTS % ep:
                            continue
                        if n % (pp * tp) or (n // (pp * tp) * tp) % ep:
                            continue
                        static, r, _, _, _ = static_gib(pp, vp, ep, tp, n)
                        if static > hw["hbm"]:
                            continue
                        flight = in_flight(pp, vp, m)
                        acts = flight * act_gib_per_layer(tp, 1)
                        lower = attnres_lower_gib(pp, vp, m, tp)
                        room = hw["hbm"] - static - lower
                        must_leave = max(0.0, acts - room) / acts if room > 0 else 1.0
                        ib = ep_ib_ms(ep, tp, hw)
                        fwd = layer_fwd_ms(tp, hw)
                        print(f"| {pp} x {vp} | {ep} | {tp} | {S} | {LAYERS / S:.1f} | {static:.0f} (r{r}) "
                              f"| {flight} ({m}) | {acts:.0f} | {lower:.1f} | {room:.0f} | {must_leave:.0%} "
                              f"| {ib:.0f} / {fwd:.0f} |")


if __name__ == "__main__":
    main()
