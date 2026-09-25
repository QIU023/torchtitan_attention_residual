"""Production model of Kimi K3 2.8T pipeline memory and step time: pp_review_optimize alone, with
pp_balance (activations parked on a peer rank) and with pp_offload (activations and the rank store
parked on host memory), on H100 and GB300 with 400 Gb/s between ranks.

Run from a checkout of pp_review4 with torch nightly:
    PYTHONPATH=<pp_review4>:. python pp_prod_model_2026-09-24.py [--quick]

A discrete-event pass runs torch's Interleaved1F1B order on every rank. A forward or backward takes
its stage's layers times the per-layer time; hops carry (1 + K) units over the sender's and the
receiver's NIC queues; parks, fetches and host copies queue on the same NIC or on the host link.
An action waits for its inputs, and a backward waits for any tensor it needs that is still in
flight back (a stall). Memory per rank is then swept from the realized times.

Memory classes per micro-batch (units of 8192 x 7168 bf16 = 0.109 GiB):
  static        bf16 params, ZeRO-1 fp32 master and momentum shards, grad double buffer, MoonEP
                buffers, 4 GiB reserve (k3_2p8t_sizing_2026-09-24.py)
  store         V4's [rows, T, D] buffer per micro-batch, first use to the rank's last backward
  hidden        stage input (receive on demand) to its backward; stage output to its backward
  grad sends    input gradients sent, pinned to the end of the step (torch runtime)
  deposits      stored blocks' gradients between the reading and the bringing stage's backward
  saved         FullAC layer inputs beyond a stage's first, and the stack a mid-stage block
                opening leaves for the stage's later layers: what pp_balance and pp_offload move
  transient     the backward recompute of one layer: the unfused AttnRes aggregation keeps fp32
                copies of the stack (about 16 units per block entry) plus about 40 units for
                attention and MoE (an estimate; a fused aggregation removes the first term)
Base (pp_review4) adds torch's persistent receive buffers, the leaf and payload copies and the
forward sends pinned to the end of the step.
"""

import argparse
import importlib.util
import os
from collections import defaultdict
from dataclasses import dataclass, field

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


memv2 = _load("memv2", "pp_memory_model_v2_2026-09-24.py")
sizing = _load("sizing", "k3_2p8t_sizing_2026-09-24.py")

GiB = 2**30
LAYERS, BLOCK, DIM, SEQ, VOCAB = 93, 12, 7168, 8192, 163840
UNIT = SEQ * DIM * 2  # bytes
MLA_LAYERS = set(range(3, 92, 4)) | {92}


@dataclass
class HW:
    name: str
    hbm_gib: float
    tflops: float  # effective BF16 on MoE GEMMs
    nic_gbps: float = 50.0  # 400 Gb/s per direction per GPU
    nic_free: float = 1.0  # share of the NIC left to PP hops and parks (EP all-to-all takes the rest)
    host_gbps: float = 50.0  # per direction
    latency_s: float = 20e-6


H100 = HW("H100", 80e9 / GiB, 450.0, nic_free=0.5, host_gbps=50.0)
GB300 = HW("GB300", 288e9 / GiB, 1100.0, nic_free=1.0, host_gbps=200.0)


@dataclass
class Layout:
    pp: int
    vp: int
    m: int
    ep: int


@dataclass
class Policy:
    """What moves off the device and when it comes back (lead = actions before the consumer)."""

    balance: dict = field(default_factory=dict)  # source rank -> (dest rank, first K forwards parked)
    balance_lead: int = 2
    offload_saved: dict = field(default_factory=dict)  # rank -> first K forwards whose saves go to host
    offload_store: bool = False
    offload_lead: int = 2
    staging_gib: float = 1.0
    early_grad_wait: bool = False


def layer_active_params(layer):
    if layer == 0:
        return 0.443e9 + 0.727e9
    attn = 0.232e9 if layer in MLA_LAYERS else 0.443e9
    return attn + 0.19e9 + 16 * 3 * 3584 * 3072


class Model:
    def __init__(self, hw: HW, lay: Layout, n_in_override=None, recipe: str = "fullac"):
        """recipe: "fullac" (each layer saves its input, recompute everything) or "report" (the K3
        report: element-wise ops recomputed, the rest saved in FP8, about 15 units per layer)."""
        self.hw, self.lay, self.recipe = hw, lay, recipe
        self.t, self.stage_layers = memv2.build(lay.pp, lay.vp, LAYERS, BLOCK)
        S = self.S = self.t.num_stages
        t = self.t
        self.k_out = {s: len(t.delta_to_send(s)) if s < S - 1 else 0 for s in range(S)}
        self.k_in = {s: self.k_out[s - 1] if s else 0 for s in range(S)}
        self.n_in = {s: len(t.cache_at_entry(s)) + self.k_in[s] for s in range(S)}
        self.commits = {s: len(t.commits_at(s)) for s in range(S)}
        self.mid_open = {
            s: any(BLOCK * b < max(self.stage_layers[s]) for b in t.commits_at(s)) for s in range(S)
        }
        self.rank_of = {s: s % lay.pp for s in range(S)}
        self.stages_of = {r: [s for s in range(S) if s % lay.pp == r] for r in range(lay.pp)}
        self.rows = {
            r: max(self.n_in[s] + self.commits[s] for s in self.stages_of[r]) for r in range(lay.pp)
        }
        flops_per_param = 2 * SEQ / (hw.tflops * 1e12)
        self.t_fwd = {}
        for s in range(S):
            params = sum(layer_active_params(l) for l in self.stage_layers[s])
            if s == S - 1:
                params += DIM * VOCAB
            self.t_fwd[s] = params * flops_per_param
        # FullAC: recompute + two grads; the report's recipe recomputes only element-wise ops
        factor = 3.0 if recipe == "fullac" else 2.2
        self.t_bwd = {s: factor * self.t_fwd[s] for s in range(S)}
        order, _ = memv2.schedule(lay.pp, lay.vp, lay.m)
        self.actions = {
            r: [
                ("F" if a.computation_type == memv2._ComputationType.FORWARD else "B", a.stage_index, a.microbatch_index)
                for a in order[r]
                if a is not None
            ]
            for r in range(lay.pp)
        }
        self.index = {r: {a: i for i, a in enumerate(self.actions[r])} for r in range(lay.pp)}
        self.static = self._static()
        self.move_order = None

    def rank_moves(self, start, end):
        """Each rank's forwards ranked by what moving their saves is worth: bytes times time held."""
        order = {}
        for r in range(self.lay.pp):
            fwds = [a for a in self.actions[r] if a[0] == "F"]

            def worth(a):
                acts, opened = self.saved_units(a[1])
                return (acts + opened) * (start[("B", a[1], a[2])] - end[a])

            order[r] = sorted(fwds, key=lambda a: -worth(a))
        self.move_order = order

    def _static(self):
        lay = self.lay
        n = 4096
        split = memv2._generate_llm_fqn_per_model_part(self.S, LAYERS, 1, 1)
        l2s = memv2.layer_to_stage_from_split(split)
        out = {}
        for r in range(lay.pp):
            layers = [l for l, st in l2s.items() if st % lay.pp == r]
            experts = sum(sizing.EXPERTS_PER_LAYER_B / lay.ep for l in layers if l)
            dense = sum(sizing.nonexpert_b(l) for l in layers)
            if r == 0:
                dense += sizing.EMBED_B + sizing.VISION_B
            if r == (self.S - 1) % lay.pp:
                dense += sizing.HEAD_B + 0.634
                experts += sizing.EXPERTS_PER_LAYER_B / lay.ep
            params = 2 * (experts + dense)
            optim = 8 * experts / (n / (lay.pp * lay.ep)) + 8 * dense / (n / lay.pp)
            redundant = (896 / lay.ep) * sizing.EXPERT_PARAMS * 6 / 1e9
            dispatch = SEQ * 16 * 3584 * 3 / 1e9
            out[r] = (params + optim + 2.0 + redundant + dispatch) * 1e9 / GiB + 4.0
        return out

    def saved_units(self, s):
        """Units checkpointing saves for one micro-batch of stage s (movable)."""
        acts = len(self.stage_layers[s]) - (1 if s else 0)
        opened = (self.n_in[s] + self.commits[s]) if (self.commits[s] and self.mid_open[s]) else 0
        return acts, opened

    fused_attnres = True

    def transient_units(self, s):
        # the unfused aggregation keeps fp32 copies of the stack; a fused kernel keeps a [T, D] slice
        agg = 0 if self.fused_attnres else 16 * (self.n_in[s] + self.commits[s] + 1)
        return agg + (40 if self.recipe == "fullac" else 20)

    def layer_saves(self, j):
        """Units a layer's checkpoint keeps besides the opened stack; the first layer's input is the stage's."""
        if self.recipe == "fullac":
            return 1 if j else 0
        return 15 + (1 if j else 0)


def layer_chunks(md, s):
    """Per layer of stage s in forward order: the units its checkpoint saves that may leave the device.

    A layer's input is saved by its checkpoint (the stage's first input is the received hidden, which
    the stage itself holds). The stack a block opening makes is the next layer's checkpoint input; it is
    counted once, with the first layer after the opening. Views of the rank store stay on the device.
    """
    layers = md.stage_layers[s]
    chunks = []
    opened_pending = 0
    for j, layer in enumerate(layers):
        units = md.layer_saves(j)
        units += opened_pending
        opened_pending = 0
        if layer % BLOCK == 0 and layer > 0 or (layer == 0 and j + 1 < len(layers)):
            opened_pending = md.n_in[s] + md.commits[s] if md.commits[s] else 0
        chunks.append(units)
    return chunks


def simulate(md: Model, scheme: str, pol: Policy | None = None, store_plan=None):
    """Returns (per-rank peak GiB, step seconds, stall seconds per rank, traffic and details)."""
    pol = pol or Policy()
    hw, lay, S = md.hw, md.lay, md.S
    pp, m = lay.pp, lay.m
    bw = hw.nic_gbps * 1e9 * hw.nic_free
    hbw = hw.host_gbps * 1e9
    nic_out, nic_in = [0.0] * pp, [0.0] * pp
    host_out, host_in = [0.0] * pp, [0.0] * pp
    traffic = defaultdict(float)

    def net(a, b, nbytes, t0):
        st = max(t0, nic_out[a], nic_in[b])
        e = st + nbytes / bw + hw.latency_s
        nic_out[a] = nic_in[b] = e
        traffic["nic"] += nbytes
        return e

    def d2h(r, nbytes, t0):
        host_out[r] = max(t0, host_out[r]) + nbytes / hbw
        traffic["host"] += nbytes
        return host_out[r]

    def h2d(r, nbytes, t0):
        host_in[r] = max(t0, host_in[r]) + nbytes / hbw
        traffic["host"] += nbytes
        return host_in[r]

    use_balance = scheme.endswith(("balance", "both"))
    use_offload = scheme.endswith(("offload", "both"))
    order = md.move_order or {r: [a for a in md.actions[r] if a[0] == "F"] for r in range(pp)}
    backend = {}  # (s, mb) -> ("remote", dest) | ("host", None)
    for r in range(pp):
        kb = pol.balance.get(r, (None, 0))[1] if use_balance else 0
        ko = pol.offload_saved.get(r, 0) if use_offload else 0
        for a in order[r][:kb]:
            backend[a[1:]] = ("remote", pol.balance[r][0])
        for a in order[r][kb:kb + ko]:
            backend[a[1:]] = ("host", None)

    arrive_f, arrive_b, start, end = {}, {}, {}, {}
    chunk_life = defaultdict(list)  # rank -> (t_alloc, t_free, bytes, cls)
    pool_ev = defaultdict(list)  # dest rank -> (t, +/- bytes)
    first_ready = {}  # (s, mb) -> ready time of the chunk the backward needs first
    pending_first = defaultdict(list)  # (rank, action index) -> [(s, mb)] to prefetch at its start
    stall = [0.0] * pp
    rank_free = [0.0] * pp
    nxt = [0] * pp
    parked = {}  # (s, mb) -> list of (units, freed_time) per layer, for the chunks that left

    def deps(a):
        kind, s, mb = a
        if kind == "F":
            return 0.0 if s == 0 else arrive_f.get((s, mb))
        if s == S - 1:
            return end.get(("F", s, mb))
        return arrive_b.get((s, mb))

    def fetch(r, s, mb, j, t_issue):
        units, freed = parked[(s, mb)][j]
        nbytes = units * UNIT
        kind, dest = backend[(s, mb)]
        if kind == "remote":
            ready = net(dest, r, nbytes, max(t_issue, freed))
        else:
            ready = h2d(r, nbytes, max(t_issue, freed))
        return ready, nbytes, kind, dest

    total = sum(len(v) for v in md.actions.values())
    for _ in range(total):
        best = None
        for r in range(pp):
            if nxt[r] >= len(md.actions[r]):
                continue
            a = md.actions[r][nxt[r]]
            d = deps(a)
            if d is None:
                continue
            t0 = max(rank_free[r], d)
            if best is None or t0 < best[0]:
                best = (t0, r, a)
        t0, r, a = best
        kind, s, mb = a
        idx = nxt[r]
        for key in pending_first.pop((r, idx), []):
            ks, kmb = key
            L = len(md.stage_layers[ks])
            ready, nbytes, bk, dest = fetch(r, ks, kmb, L - 1, t0)
            first_ready[key] = (t0, ready)
        L = len(md.stage_layers[s])
        if kind == "F":
            dur = md.t_fwd[s]
            tl = dur / L
            chunks = layer_chunks(md, s)
            if (s, mb) in backend:
                bk, dest = backend[(s, mb)]
                rec = []
                for j, units in enumerate(chunks):
                    t_saved = t0 + j * tl
                    if units == 0:
                        rec.append((0, t_saved))
                        continue
                    nbytes = units * UNIT
                    freed = net(r, dest, nbytes, t_saved) if bk == "remote" else d2h(r, nbytes, t_saved)
                    chunk_life[r].append((t_saved, freed, nbytes, "saved"))
                    if bk == "remote":
                        pool_ev[dest].append((freed, nbytes))
                    rec.append((units, freed))
                parked[(s, mb)] = rec
                b_idx = md.index[r][("B", s, mb)]
                trigger = max(b_idx - pol.balance_lead if bk == "remote" else b_idx - pol.offload_lead, idx + 1)
                pending_first[(r, trigger)].append((s, mb))
            else:
                for j, units in enumerate(chunks):
                    if units:
                        chunk_life[r].append((t0 + j * tl, None, units * UNIT, ("keep", s, mb, j)))
            start[a], end[a] = t0, t0 + dur
            rank_free[r] = t0 + dur
            if s < S - 1:
                arrive_f[(s + 1, mb)] = net(r, md.rank_of[s + 1], (1 + md.k_out[s]) * UNIT, t0 + dur)
        else:
            dur = md.t_bwd[s]
            tl = dur / L
            acc = 0.0
            if (s, mb) in parked:
                rec = parked[(s, mb)]
                bk, dest = backend[(s, mb)]
                for j in range(L - 1, -1, -1):
                    units, freed = rec[j]
                    need = t0 + (L - 1 - j) * tl + acc
                    if units == 0:
                        continue
                    if j == L - 1 and (s, mb) in first_ready:
                        issued, ready = first_ready[(s, mb)]
                    else:
                        issued = t0 + max(0, L - 2 - j) * tl + acc
                        ready, _, _, _ = fetch(r, s, mb, j, issued)
                    if ready > need:
                        acc += ready - need
                        need = ready
                    back_free = need + tl
                    chunk_life[r].append((issued, back_free, units * UNIT, "saved"))
                    if bk == "remote":
                        pool_ev[dest].append((ready, -units * UNIT))
            stall[r] += acc
            start[a], end[a] = t0, t0 + dur + acc
            rank_free[r] = t0 + dur + acc
            if s > 0:
                arrive_b[(s - 1, mb)] = net(r, md.rank_of[s - 1], (1 + md.k_in[s]) * UNIT, t0 + dur + acc)
        nxt[r] += 1
    step = max(end.values())
    # the kept chunks live from their save to their layer's backward
    for r in range(pp):
        fixed = []
        for t_a, t_f, nbytes, cls in chunk_life[r]:
            if isinstance(cls, tuple):
                _, s, mb, j = cls
                L = len(md.stage_layers[s])
                bs = start[("B", s, mb)]
                t_f = bs + (end[("B", s, mb)] - bs) * (L - j) / L
                fixed.append((t_a, t_f, nbytes, "saved"))
            else:
                fixed.append((t_a, t_f, nbytes, cls))
        chunk_life[r] = fixed
    store_resident = {}
    if use_offload and pol.offload_store and store_plan is not None:
        store_resident, extra_stall, store_bytes = store_plan(md, start, end, arrive_f, pol, bool(any(pol.offload_saved.values())))
        traffic["host"] += store_bytes
        for r in range(pp):
            stall[r] += extra_stall[r]

    # ---- memory sweep ----
    events = {r: [] for r in range(pp)}

    def grad_free(s, mb, be):
        if not pol.early_grad_wait:
            return step
        rr, prev = md.rank_of[s], md.rank_of[s - 1]
        consumed = end[("B", s - 1, mb)]
        later = [
            start[x] for x in md.actions[rr]
            if x[0] == "F" and x[1] > 0 and md.rank_of[x[1] - 1] == prev
            and start[("F", x[1] - 1, x[2])] >= consumed and start[x] >= be
        ]
        return min(later) if later else step

    cls_now = ['other']

    def alloc(r, t_a, t_f, units_or_bytes, is_bytes=False):
        nbytes = units_or_bytes if is_bytes else units_or_bytes * UNIT
        if nbytes <= 0 or t_f <= t_a:
            return
        events[r].append((t_a, nbytes, cls_now[0]))
        events[r].append((t_f, -nbytes, cls_now[0]))

    for r in range(pp):
        cls_now[0] = 'saved'
        for t_a, t_f, nbytes, _ in chunk_life[r]:
            alloc(r, t_a, t_f, nbytes, True)
        for mb in range(m):
            f_first = min(start[("F", s, mb)] for s in md.stages_of[r])
            f_last = max(end[("F", s, mb)] for s in md.stages_of[r])
            b_last = max(end[("B", s, mb)] for s in md.stages_of[r])
            if scheme != "base":
                cls_now[0] = 'store'
                for t_a, t_f in store_resident.get((r, mb), [(f_first, b_last)]):
                    alloc(r, t_a, t_f, md.rows[r])
            for s in md.stages_of[r]:
                fs, fe = start[("F", s, mb)], end[("F", s, mb)]
                bs, be = start[("B", s, mb)], end[("B", s, mb)]
                cls_now[0] = 'transient'
                alloc(r, bs, be, md.transient_units(s))
                if scheme == "base":
                    cls_now[0] = 'copies'
                    if s > 0 and md.n_in[s]:
                        alloc(r, fs, be, md.n_in[s])
                    if md.commits[s]:
                        stop = max(f_last, be) if md.mid_open[s] else f_last
                        alloc(r, fs, stop, md.n_in[s] + md.commits[s])
                    if s < S - 1:
                        alloc(r, fe, step, 1 + md.k_out[s])
                    if s > 0:
                        cls_now[0] = 'grad sends'
                        alloc(r, be, grad_free(s, mb, be), 1 + md.k_in[s])
                else:
                    cls_now[0] = 'hidden'
                    if s > 0:
                        alloc(r, arrive_f[(s, mb)], be, 1)
                    if s < S - 1:
                        alloc(r, fe, be, 1)
                    if s > 0:
                        cls_now[0] = 'grad sends'
                        alloc(r, be, grad_free(s, mb, be), 1 + md.k_in[s])
                cls_now[0] = 'deposits'
                for blk in md.t.cache_at_entry(s):
                    bringer = next(
                        q for q in md.stages_of[r]
                        if blk in md.t.commits_at(q) or (q and blk in md.t.delta_to_send(q - 1))
                    )
                    alloc(r, be, end[("B", bringer, mb)], 1)
    peaks, compositions = {}, {}
    for r in range(pp):
        base_bytes = md.static[r] * GiB
        if scheme == "base":
            base_bytes += sum(
                m * ((1 + md.k_in[s]) if s else 0) + m * ((1 + md.k_out[s]) if s < S - 1 else 0)
                for s in md.stages_of[r]
            ) * UNIT
        if use_balance:
            if r in pol.balance and pol.balance[r][1]:
                base_bytes += pol.staging_gib * GiB
            cur = pk = 0
            for _, d in sorted(pool_ev.get(r, [])):
                cur += d
                pk = max(pk, cur)
            base_bytes += pk  # the destination's pool, registered for the whole run
        cur = peak = base_bytes
        comp = defaultdict(float)
        peak_comp, peak_t = {}, 0.0
        for tt, delta, cls in sorted(events[r], key=lambda e: (e[0], e[1])):
            cur += delta
            comp[cls] += delta
            if cur > peak:
                peak, peak_comp, peak_t = cur, dict(comp), tt
        peaks[r] = peak / GiB
        compositions[r] = (peak_t / step, {k: v / GiB for k, v in peak_comp.items() if v > 1e6}, base_bytes / GiB)
    traffic["composition"] = compositions
    traffic["times"] = (start, end)
    return peaks, step, stall, dict(traffic)


def max_parked(moves, md, src, dest):
    """The pool a destination must hold for one source: the most bytes parked at once."""
    ev = []
    for (s, mb), mv in moves.items():
        if mv["kind"] == "balance" and mv["dest"] == dest and md.rank_of[s] == src:
            ev.append((mv["freed"], mv["nbytes"]))
            ev.append((mv["back_ready"], -mv["nbytes"]))
    cur = peak = 0
    for _, d in sorted(ev):
        cur += d
        peak = max(peak, cur)
    return peak


def fmt(peaks):
    return ", ".join(f"{peaks[r]:.1f}" for r in sorted(peaks))


def store_windows(md, start, end, arrive_f, pol, shared):
    """Park each micro-batch's store buffer on host between uses when the round trip fits.

    Uses on a rank: a later stage's delta landing in its rows (from the transfer's start), each
    stage's forward (to its payload send's arrival), each stage's backward (the recompute reads
    its stack). Candidate windows are taken longest first; one is kept only if, on the rank's host
    queues as they stand, the copy out finishes and the copy back (started at the action
    `offload_lead` before the next use, or once the copy out is done) lands before that use.
    """
    hbw = md.hw.host_gbps * 1e9 * (0.5 if shared else 1.0)
    bw = md.hw.nic_gbps * 1e9 * md.hw.nic_free
    resident, stall = {}, [0.0] * md.lay.pp
    moved = 0.0
    for r in range(md.lay.pp):
        nbytes = md.rows[r] * UNIT
        copy = nbytes / hbw
        busy_out, busy_in = [], []  # (start, end) intervals already booked on the host queues

        def book(busy, t0, dur):
            t = t0
            for a, b in sorted(busy):
                if t + dur <= a:
                    break
                if b > t:
                    t = b
            busy.append((t, t + dur))
            return t + dur

        cands = []
        uses_of = {}
        for mb in range(md.lay.m):
            uses = []
            for s in md.stages_of[r]:
                if s > 0 and s != md.stages_of[r][0]:
                    land = arrive_f[(s, mb)] - (1 + md.k_in[s]) * UNIT / bw
                    uses.append((land, arrive_f[(s, mb)], ("F", s, mb)))
                send_done = arrive_f.get((s + 1, mb), end[("F", s, mb)])
                uses.append((start[("F", s, mb)], max(end[("F", s, mb)], send_done), ("F", s, mb)))
                uses.append((start[("B", s, mb)], end[("B", s, mb)], ("B", s, mb)))
            uses.sort(key=lambda u: u[0])
            uses_of[mb] = uses
            for i, ((a0, a1, _), (b0, b1, act)) in enumerate(zip(uses, uses[1:])):
                if b0 - a1 > 2 * copy:
                    cands.append((b0 - a1, mb, i))
        taken = defaultdict(list)
        for gap, mb, i in sorted(cands, reverse=True):
            a1 = uses_of[mb][i][1]
            b0, act = uses_of[mb][i + 1][0], uses_of[mb][i + 1][2]
            t_out = book(busy_out, a1, copy)
            idx = md.index[r][act]
            trigger = start[md.actions[r][max(0, idx - pol.offload_lead)]]
            t_in = book(busy_in, max(trigger, t_out), copy)
            if t_in > b0:
                busy_out.pop()
                busy_in.pop()
                continue
            taken[mb].append((a1, t_out, t_in - copy))
            moved += 2 * nbytes
        for mb, uses in uses_of.items():
            spans, cur = [], uses[0][0]
            for a1, freed, back in sorted(taken[mb]):
                spans.append((cur, freed))
                cur = back
            spans.append((cur, uses[-1][1]))
            resident[(r, mb)] = spans
    return resident, stall, moved


def run(md, scheme, pol=None):
    peaks, step, stall, traffic = simulate(md, scheme, pol, store_windows)
    return peaks, step, stall, traffic


def search_offload(md, store, v4_step, eps, base_pol=None):
    """Per rank, the most forwards whose saves go to host without the step growing past eps."""
    best = None
    for lead in (1, 2, 3, 4, 6, 8):
        pol = Policy(offload_lead=lead, offload_store=store, early_grad_wait=True)
        if base_pol is not None:
            pol.balance, pol.balance_lead = dict(base_pol.balance), base_pol.balance_lead
        scheme = "v4+both" if base_pol is not None else "v4+offload"
        n_fwd = {r: sum(1 for a in md.actions[r] if a[0] == "F") for r in range(md.lay.pp)}
        peaks, step, _, _ = run(md, scheme, pol)
        for r in sorted(peaks, key=lambda q: -peaks[q]):
            kb = pol.balance.get(r, (None, 0))[1]
            best_k, best_peak = 0, peaks[r]
            for k in range(2, n_fwd[r] - kb + 1, 2):
                pol.offload_saved[r] = k
                pk, st, _, _ = run(md, scheme, pol)
                if st > v4_step * (1 + eps):
                    break
                if pk[r] < best_peak - 1e-6:
                    best_k, best_peak = k, pk[r]
            pol.offload_saved[r] = best_k
        peaks, step, stall, traffic = run(md, scheme, pol)
        if step <= v4_step * (1 + eps) and (best is None or max(peaks.values()) < max(best[1].values()) - 1e-6):
            best = (pol, peaks, step, stall, traffic)
    return best


def search_balance(md, v4_step, eps, pol0=None):
    """Heaviest ranks park on the lightest, pairwise; per source the forwards parked, then the lead."""
    scheme = "v4+both" if pol0 is not None and (any(pol0.offload_saved.values()) or pol0.offload_store) else "v4+balance"
    start_pol = pol0 or Policy(early_grad_wait=True)
    peaks0, _, _, _ = run(md, scheme if pol0 else "v4", start_pol)
    order = sorted(peaks0, key=lambda q: -peaks0[q])
    best = None
    for n_src in range(1, md.lay.pp // 2 + 1):
        srcs, dests = order[:n_src], list(reversed(order))[:n_src]
        for lead in (1, 2, 3, 4, 6, 8):
            pol = Policy(
                balance={src: (dst, 0) for src, dst in zip(srcs, dests)},
                balance_lead=lead,
                offload_saved=dict(start_pol.offload_saved),
                offload_store=start_pol.offload_store,
                offload_lead=start_pol.offload_lead,
                early_grad_wait=True,
            )
            n_fwd = {r: sum(1 for a in md.actions[r] if a[0] == "F") for r in range(md.lay.pp)}
            for src, dst in zip(srcs, dests):
                ko = pol.offload_saved.get(src, 0)
                best_k, best_worst = 0, None
                for k in range(0, n_fwd[src] - ko + 1, 2):
                    pol.balance[src] = (dst, k)
                    pk, st, _, _ = run(md, scheme if pol0 else "v4+balance", pol)
                    if st > v4_step * (1 + eps):
                        break
                    worst = max(pk[src], pk[dst])
                    if best_worst is None or worst < best_worst - 1e-6:
                        best_k, best_worst = k, worst
                pol.balance[src] = (dst, best_k)
            peaks, step, stall, traffic = run(md, scheme if pol0 else "v4+balance", pol)
            if step <= v4_step * (1 + eps) and (best is None or max(peaks.values()) < max(best[1].values()) - 1e-6):
                best = (pol, peaks, step, stall, traffic)
    return best


def lead_sweep(md, pol, scheme, field_name, leads=(1, 2, 4, 8, 16)):
    """Step time and peaks as the lead moves: too short stalls the backward, too long holds memory."""
    rows = []
    for lead in leads:
        p = Policy(**{**pol.__dict__})
        p.balance = dict(pol.balance)
        p.offload_saved = dict(pol.offload_saved)
        setattr(p, field_name, lead)
        peaks, step, stall, _ = run(md, scheme, p)
        rows.append((lead, max(peaks.values()), sum(peaks.values()) / len(peaks), step, max(stall)))
    return rows


def report(label, hw, lay, eps=0.01, recipe="fullac"):
    md = Model(hw, lay, recipe=recipe)
    _, _, _, tr = run(md, "v4", Policy(early_grad_wait=True))
    md.rank_moves(*tr["times"])
    print(f"\n=== {label} ({recipe}): pp{lay.pp} x vp{lay.vp}, M={lay.m}, EP{lay.ep}; HBM {hw.hbm_gib:.1f} GiB; "
          f"NIC {hw.nic_gbps * 8:.0f} Gb/s x {hw.nic_free} for PP and parks; host {hw.host_gbps:.0f} GB/s; "
          f"{hw.tflops:.0f} TFLOPS")
    base = run(md, "base")
    v4_plain = run(md, "v4")
    v4 = run(md, "v4", Policy(early_grad_wait=True))
    v4_step = v4[1]
    rows = [("pp_review4", base, None), ("V4", v4_plain, None), ("V4 + early grad wait", v4, Policy(early_grad_wait=True))]
    bal = search_balance(md, v4_step, eps)
    rows.append(("V4 + balance", bal[1:], bal[0]))
    off_saved = search_offload(md, False, v4_step, eps)
    rows.append(("V4 + offload (saves)", off_saved[1:], off_saved[0]))
    off_all = search_offload(md, True, v4_step, eps)
    rows.append(("V4 + offload (saves + store blocks, not in the report)", off_all[1:], off_all[0]))
    both = search_balance(md, v4_step, eps, pol0=off_saved[0])
    rows.append(("V4 + offload + balance", both[1:], both[0]))
    print("| scheme | " + " | ".join(f"r{r}" for r in range(lay.pp)) + " | max | mean | step s | worst stall s | NIC GB | host GB | knobs |")
    print("|---|" + "---:|" * lay.pp + "---:|---:|---:|---:|---:|---:|---|")
    for name, (peaks, step, stall, traffic), pol in rows:
        knobs = ""
        if pol is not None:
            if pol.balance:
                knobs += "balance " + ", ".join(f"r{s}->r{d} K{k}" for s, (d, k) in pol.balance.items() if k) + f" lead {pol.balance_lead}; "
            if any(pol.offload_saved.values()) or pol.offload_store:
                knobs += "offload " + ", ".join(f"r{r} K{k}" for r, k in pol.offload_saved.items() if k)
                knobs += (" + store" if pol.offload_store else "") + f" lead {pol.offload_lead}"
        print(f"| {name} | " + " | ".join(f"{peaks[r]:.1f}" for r in range(lay.pp)) +
              f" | {max(peaks.values()):.1f} | {sum(peaks.values()) / lay.pp:.1f} | {step:.2f} | {max(stall):.3f} "
              f"| {traffic.get('nic', 0) / 1e9:.0f} | {traffic.get('host', 0) / 1e9:.0f} | {knobs} |")
    comp = both[4]["composition"]
    print("at each rank's peak with offload + balance (fraction of step; GiB by class): " + "; ".join(
        f"r{r} {comp[r][0]:.2f}: fixed {comp[r][2]:.1f} " + " ".join(f"{k} {v:.1f}" for k, v in sorted(comp[r][1].items()))
        for r in range(lay.pp)))
    # timing sensitivity of the chosen combination
    pol = both[0]
    print("lead sweep, balance (lead, max GiB, mean GiB, step s, worst stall s):",
          [(l, round(a, 1), round(b, 1), round(c, 2), round(d, 3)) for l, a, b, c, d in lead_sweep(md, pol, "v4+both", "balance_lead")])
    print("lead sweep, offload:",
          [(l, round(a, 1), round(b, 1), round(c, 2), round(d, 3)) for l, a, b, c, d in lead_sweep(md, pol, "v4+both", "offload_lead")])
    return md, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--recipe", default="fullac")
    ap.add_argument("--only", default="")
    ap.add_argument("--host-gbps", type=float, default=0.0, help="override the host link, GB/s per direction")
    args = ap.parse_args()
    if args.host_gbps:
        H100.host_gbps = GB300.host_gbps = args.host_gbps
    cases = [
        ("h100a", "H100, 2.8T", H100, Layout(8, 4, 16, 64)),
        ("h100b", "H100, 2.8T", H100, Layout(16, 2, 32, 32)),
        ("gb300a", "GB300, 2.8T", GB300, Layout(4, 4, 16, 64)),
        ("gb300b", "GB300, 2.8T", GB300, Layout(2, 8, 16, 64)),
    ]
    for key, label, hw, lay in cases:
        if args.only and key not in args.only.split(","):
            continue
        if args.quick and key != "h100a":
            continue
        report(label, hw, lay, recipe=args.recipe)


if __name__ == "__main__":
    main()
