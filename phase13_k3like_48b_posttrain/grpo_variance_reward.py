"""A reward with VARIANCE inside each GRPO group, so the actor actually moves.

Why this exists rather than a real task's reward: with gsm8k exact match on a
randomly initialised fixture every reward is 0, so GRPO's group normalisation gives
every sample an advantage of 0, the policy gradient is 0, and ``actor/grad_norm`` is
exactly 0.0 -- measured. With the weights never changing, a working weight sync and a
no-op one produce identical ``rollout_probs_diff``, so the sync is untestable. Merely
making the reward NON-ZERO does not help either: a constant reward normalises to zero
advantage just as well. It has to VARY within the group.

This is deliberately not a quality signal. It is a probe whose only job is to move
the policy in a reproducible direction so the sync becomes observable. Anything read
off it about model quality would be meaningless, which is why the reward is a
function of surface form and says so.

Determinism matters for a differential check -- run with the sync on and off and the
difference must be attributable to the sync, not to reward noise -- so the value is a
hash of the response text rather than anything sampled here.
"""

from __future__ import annotations

import hashlib


def surface_form_reward(
    data_source=None,
    solution_str: str = "",
    ground_truth=None,
    extra_info=None,
    **kwargs,
) -> float:
    """A deterministic pseudo-random reward in [0, 1), keyed on the response text.

    Two responses in the same GRPO group differ in text, so they differ in reward,
    so the group's advantages are not all equal and the gradient is not zero. Same
    text always gives the same number, which is what makes an A/B against the sync
    interpretable.
    """
    digest = hashlib.sha256((solution_str or "").encode("utf-8")).digest()
    # 16 bits is ample spread for a group of 2-8 samples and keeps the value exact
    # in float32, so the reward itself contributes no rounding to the comparison.
    return int.from_bytes(digest[:2], "big") / 65536.0
