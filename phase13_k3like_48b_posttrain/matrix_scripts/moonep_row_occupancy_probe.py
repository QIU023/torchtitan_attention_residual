"""Cell 9 of the MoonEP plan: which rows of the [E + B] table actually receive tokens.

Run it in place of torchtitan.train, with the same flags:

    torchrun --nproc_per_node=4 moonep_row_occupancy_probe.py \
      --module kimi_k3 --config kimi_k3_debugmodel --debug.seed 42 --debug.deterministic \
      --training.steps 1 --parallelism.expert_parallel_degree 4 ...

It wraps MoonEPTokenDispatcher.dispatch, and for the first N dispatches of the run
prints the non-empty rows of cu_seqlens and the plan's experts_to_copy, then checks
the one property the table layout rests on: a row in [0, E) receives tokens only on
the rank that owns that expert, and a row in [E, E + B) only when the plan filled
that slot. If that holds on real moonep, the [E + B] table can be compacted to the
rank's own E / R rows plus its B slots (audit of 2026-09-17, MOONEP_TEST_PLAN).

A violation is printed as VIOLATION and the run is left to continue, so one step
reports every layer rather than dying on the first.
"""

from __future__ import annotations

import os
import runpy
import sys

import torch

MAX_REPORTS = int(os.environ.get("MOONEP_PROBE_LAYERS", "4"))


def _install() -> None:
    from torchtitan.models.kimi_k3.moon_ep_dispatcher import MoonEPTokenDispatcher

    original = MoonEPTokenDispatcher.dispatch
    state = {"seen": 0}

    def dispatch(self, x_TD, topk_scores_TK, topk_expert_ids_TK, counts_E):
        out = original(self, x_TD, topk_scores_TK, topk_expert_ids_TK, counts_E)
        if self.ep_mesh is None or state["seen"] >= MAX_REPORTS:
            return out
        state["seen"] += 1
        hidden, num_tokens_per_row, metadata = out
        rank, size = self.ep_mesh.get_local_rank(), self.ep_mesh.size()
        E = self.num_experts
        local = E // size
        home_lo, home_hi = rank * local, (rank + 1) * local
        rows = num_tokens_per_row.detach().to("cpu").tolist()
        B = len(rows) - E
        to_copy = metadata.plan.experts_to_copy[rank].detach().to("cpu").tolist()

        filled = [b for b, e in enumerate(to_copy) if e >= 0]
        home_rows = {r: n for r, n in enumerate(rows[:E]) if n}
        slot_rows = {b: n for b, n in enumerate(rows[E:]) if n}
        foreign = {r: n for r, n in home_rows.items() if not home_lo <= r < home_hi}
        empty_slots = {b: n for b, n in slot_rows.items() if to_copy[b] < 0}

        tag = f"[moonep-probe rank {rank} dispatch {state['seen']}]"
        print(
            f"{tag} E={E} B={B} local={local} home rows [{home_lo},{home_hi}) "
            f"padded rows received={sum(rows)} (this rank's share of "
            f"R x S x K, padded per VM group)",
            flush=True,
        )
        print(
            f"{tag} own home rows used {len(home_rows) - len(foreign)}/{local}, "
            f"slots filled by plan {len(filled)}/{B} {filled}, "
            f"slot rows with tokens {sorted(slot_rows)}",
            flush=True,
        )
        print(f"{tag} experts_to_copy[{rank}]={to_copy}", flush=True)
        if foreign:
            print(
                f"{tag} VIOLATION foreign home rows carry tokens: {foreign} "
                "(the table cannot be compacted as the audit assumed)",
                flush=True,
            )
        if empty_slots:
            print(
                f"{tag} VIOLATION rows in unplanned slots carry tokens: {empty_slots}",
                flush=True,
            )
        if not foreign and not empty_slots:
            useful = local + B
            print(
                f"{tag} OK only own {local} home rows and planned slots receive "
                f"tokens: {useful} of {E + B} rows are useful "
                f"({(E + B) / useful:.1f}x table)",
                flush=True,
            )
        return out

    MoonEPTokenDispatcher.dispatch = dispatch


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("moonep_row_occupancy_probe needs the GPUs the run uses")
    _install()
    runpy.run_module("torchtitan.train", run_name="__main__")
