"""Reorder a pipeline IR so vision work lands in stalls (report 5.2.3).

Pure functions over an action list. No torch, no distributed, no hardware: the input
is a per-rank list of actions and the output is another one, so the whole thing is
testable offline and its result is a property of the schedule rather than of a box.

## Why a reorder at all

Measured, pp8 x Interleaved1F1B, 16 virtual stages: making the ViT a pipeline STAGE
changes the bubble count not at all -- 2019 with DEP, 2019 without. A stage's actions
sit on the critical path, and the schedule does not know one of its stages is an
encoder. So "hidden within pipeline bubbles" needs the action list changed, not just
the module placement.

## What the report constrains, and what it leaves free

"The ViT forward passes of the first PP micro-batches are executed synchronously
upfront, the remaining forward passes are scheduled into pipeline bubbles, and the
backward passes are handled analogously."

Constrained: micro-batch m's vision forward must precede the text forward that
consumes m. Leading micro-batches have no slack -- their text forwards are at the
very start -- so their vision forwards stay put. Later ones have slack that grows
with m.

Left free by the report: how many leading micro-batches count as "first", and how far
to defer the rest. Both are parameters here, and the honest way to set them is a
search with the simulator's bubble count as the oracle, not a rule invented and then
described as the report's.

## Cost model, and its limit

``bubble_cost`` weights each idle slot. torch's simulator counts unit slots, so a ViT
forward costs the same as one text stage's forward -- and those differ by orders of
magnitude in both directions depending on the configuration. Measured from the
configs: at ``report_arch_pp8vp4`` one ViT forward is 25.2 text layers, at the real
``2p8t_vl`` with one 1024-patch image it is 0.057 of one layer. So the unit-slot count
answers "does the schedule place vision work in stalls" and nothing about how much
time that hides. A share requires the visual token budget per micro-batch, which is a
data parameter and cannot be derived from a model config.
"""

from __future__ import annotations

FORWARD_NAMES = ("FORWARD",)
BACKWARD_NAMES = ("FULL_BACKWARD", "BACKWARD_INPUT", "BACKWARD_WEIGHT")


def _kind(action) -> str:
    return str(getattr(action, "computation_type", ""))


def is_forward(action) -> bool:
    k = _kind(action)
    return any(n == k or k.endswith(n) for n in FORWARD_NAMES)


def is_backward(action) -> bool:
    k = _kind(action)
    return any(n in k for n in BACKWARD_NAMES)


def defer_vision_forwards(
    order: dict[int, list],
    vision_stages: set[int],
    *,
    keep_first: int,
    lookahead: int,
) -> dict[int, list]:
    """Move late micro-batches' vision FORWARDS later in their rank's list.

    ``keep_first`` leading micro-batches are left where they are -- the report's
    "executed synchronously upfront", and the ones whose text forwards have no slack.
    Every other vision forward is moved ``lookahead`` positions later among the
    actions its own rank already runs.

    Deferring is what can fill a stall: the action stays on the same rank and the
    same dependency edges, so the lowering still inserts the same sends and receives,
    and the simulator can be asked whether the stall count went down. Nothing here
    reorders across ranks, and nothing changes which stage owns what -- so a
    reordering that does not help cannot break correctness either, which is why this
    is safe to search over.
    """
    if lookahead <= 0:
        return {r: list(a) for r, a in order.items()}

    out: dict[int, list] = {}
    for rank, actions in order.items():
        moved: list[tuple[int, object]] = []
        kept: list = []
        for a in actions:
            if (
                a is not None
                and getattr(a, "stage_index", None) in vision_stages
                and is_forward(a)
                and (getattr(a, "microbatch_index", 0) or 0) >= keep_first
            ):
                moved.append((len(kept) + lookahead, a))
            else:
                kept.append(a)
        # Insert from the back so earlier insertions do not shift later targets.
        for target, a in sorted(moved, key=lambda t: -t[0]):
            kept.insert(min(target, len(kept)), a)
        out[rank] = kept
    return out


def hoist_vision_backwards(
    order: dict[int, list],
    vision_stages: set[int],
    *,
    keep_last: int,
    lookahead: int,
) -> dict[int, list]:
    """The backward mirror: move EARLY micro-batches' vision backwards EARLIER.

    "the backward passes are handled analogously", and the analogy inverts the
    direction. Gradient flows text -> vision, so micro-batch m's vision backward can
    run as soon as m's text backward is done. The LAST micro-batches' text backwards
    finish at the very end, so those vision backwards have no slack and stay put;
    the EARLY ones can be pulled forward into stalls.

    Deferring them, which is what the first version of this did, is a no-op: the
    vision backwards already sit at the end of the list, so pushing them later leaves
    them exactly where they were. The offline test caught that by reporting identical
    positions before and after -- which is why the test asserts movement rather than
    just preservation.

    ``keep_last`` counts from the highest micro-batch index present.
    """
    if lookahead <= 0:
        return {r: list(a) for r, a in order.items()}

    highest = max(
        (
            (getattr(a, "microbatch_index", 0) or 0)
            for acts in order.values()
            for a in acts
            if a is not None
        ),
        default=0,
    )
    cutoff = highest - keep_last
    out: dict[int, list] = {}
    for rank, actions in order.items():
        moved: list[tuple[int, object]] = []
        kept: list = []
        for a in actions:
            if (
                a is not None
                and getattr(a, "stage_index", None) in vision_stages
                and is_backward(a)
                and (getattr(a, "microbatch_index", 0) or 0) <= cutoff
            ):
                # Pull EARLIER: target is measured back from where it sits now.
                moved.append((max(0, len(kept) - lookahead), a))
            else:
                kept.append(a)
        # Insert from the front so earlier insertions do not displace later targets
        # past each other; the relative order of the moved actions is preserved.
        for target, a in sorted(moved, key=lambda t: t[0]):
            kept.insert(min(target, len(kept)), a)
        out[rank] = kept
    return out


def action_multiset(order: dict[int, list]) -> dict:
    """A reorder must PRESERVE the action set. Anything else is a different schedule,
    not a reordering of this one, and would make a bubble reduction meaningless."""
    counts: dict[str, int] = {}
    for actions in order.values():
        for a in actions:
            if a is None:
                continue
            key = (
                f"{getattr(a, 'stage_index', '?')}|{_kind(a)}|"
                f"{getattr(a, 'microbatch_index', '?')}"
            )
            counts[key] = counts.get(key, 0) + 1
    return counts
