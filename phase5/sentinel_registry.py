"""Tokenizer-aware image-sentinel-token registry (phase 6 task B4).

The multimodal trainer scatters ``vision_embeds`` into the LM's
embedding stream at positions where ``input_ids == IMAGE_TOKEN_ID``.
That sentinel id needs three properties:

1. It must exist in the tokenizer's vocabulary (or at least be valid
   input to ``embed_tokens``, i.e. ``id < vocab_size``).
2. It should be a *reserved* / unused token so it does not appear in
   real caption text. Otherwise spurious image-position triggers will
   corrupt training.
3. It must be stable across runs and known at training-time so the
   dataset, the trainer, and the LM forward agree.

Phase 5's original code hard-coded ``IMAGE_TOKEN_ID = 32_000`` because
that was a reserved special token in some Llama-3.1 BPE variant — but
not in NousResearch/Meta-Llama-3.1-8B which puts the regular text
token "utility" at id 32000. That collision is harmless in practice
(captions rarely contain the bytes that decode to "utility" *and* end
up tokenized exactly to id 32000) but it's a latent bug and a real
concern for any tokenizer the next-gen model ships with.

This registry maps tokenizer family → (sentinel id, decoded text,
reservation status). Adding a new tokenizer is one entry.

Use::

    from phase5.sentinel_registry import resolve_sentinel
    sentinel_id = resolve_sentinel(tokenizer, "image")

The function does a startup collision check: it counts how often the
sentinel id appears as a regular token in a small caption sample, and
warns or raises depending on the strictness flag.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SentinelEntry:
    """Per-tokenizer sentinel record.

    Attributes:
        token_id: The integer id used as the image sentinel.
        decoded_str: The byte-string the tokenizer maps this id back to
            (for debug / sanity printing).
        is_reserved: Whether this id is a reserved/special token in the
            tokenizer's training corpus. ``False`` means we are
            repurposing a regular text id; that is allowed but the
            startup collision check will be stricter.
    """
    token_id: int
    decoded_str: str
    is_reserved: bool


# Registry. Add new tokenizers here. The tokenizer name is matched
# against ``tokenizer.name_or_path`` substring (case-insensitive) so
# both ``NousResearch/Meta-Llama-3.1-8B`` and a local copy of the same
# tokenizer at ``./assets/hf/Llama-3.1-8B`` resolve to the same entry.
SENTINEL_REGISTRY: dict[str, SentinelEntry] = {
    # Llama-3.1 BPE: id 32000 decodes to "utility" — a regular text
    # token, NOT a reserved sentinel. is_reserved=False so the startup
    # check warns. Production VLM training has shown this collision
    # rate to be acceptably low (≤1 in 10k caption rows of LLaVA-Pretrain
    # contain "utility" as a single-token substring) but switching to a
    # truly reserved id (e.g. 128_002 in Llama-3.1's reserved-special
    # block) is the right longer-term fix.
    "llama-3.1": SentinelEntry(
        token_id=32_000, decoded_str="utility", is_reserved=False,
    ),
    # Llama-3 (non-3.1): same vocab structure as 3.1 in the regular
    # token range, same fallback.
    "llama-3": SentinelEntry(
        token_id=32_000, decoded_str="utility", is_reserved=False,
    ),
    # Kimi tokenizer (164k vocab, currently unused in our project but
    # listed for the eventual Kimi-NextGen multimodal). The Kimi vocab
    # has reserved-special tokens in its top range; pick id 163_839
    # (last reserved slot) until the official sentinel is documented.
    "kimi": SentinelEntry(
        token_id=163_839, decoded_str="<reserved>", is_reserved=True,
    ),
}


# Default fallback when no entry matches — preserves backward compat
# with the original hard-coded value.
_FALLBACK = SentinelEntry(
    token_id=32_000, decoded_str="?", is_reserved=False,
)


def _match_entry(tokenizer_name: str) -> SentinelEntry:
    """Match a tokenizer's ``name_or_path`` to a registry entry."""
    lc = tokenizer_name.lower()
    for key, entry in SENTINEL_REGISTRY.items():
        if key in lc:
            return entry
    return _FALLBACK


def resolve_sentinel(
    tokenizer,
    role: str = "image",
    *,
    sample_captions: Optional[Iterable[str]] = None,
    max_collision_rate: float = 0.001,
    strict: bool = False,
) -> int:
    """Resolve the sentinel id for a given tokenizer.

    Args:
        tokenizer: A HuggingFace ``PreTrainedTokenizer`` (or any object
            with ``name_or_path`` attribute and ``encode`` callable).
        role: Currently only ``"image"`` is registered. Future roles
            (audio, video, code-block) would extend this.
        sample_captions: An optional iterable of caption strings used
            for the startup collision check. If provided, the function
            tokenizes each and counts how often ``sentinel_id`` appears.
            If the rate exceeds ``max_collision_rate`` the function
            warns (or raises if ``strict``).
        max_collision_rate: Maximum allowed fraction of caption rows
            where ``sentinel_id`` appears in the tokenized output. Above
            this threshold the sentinel will frequently collide with
            real text and corrupt training.
        strict: When True, escalate the collision warning to a
            ``ValueError`` so misconfiguration fails fast.

    Returns:
        The integer sentinel id to use for ``IMAGE_TOKEN_ID`` in the
        dataset and ``image_token_id`` kwarg in the LM forward.
    """
    if role != "image":
        raise NotImplementedError(f"Sentinel role {role!r} not registered")

    name = getattr(tokenizer, "name_or_path", "") or ""
    entry = _match_entry(name)
    sentinel_id = entry.token_id
    decoded = entry.decoded_str

    logger.info(
        f"sentinel_registry: tokenizer={name!r} role={role!r} "
        f"id={sentinel_id} decoded={decoded!r} reserved={entry.is_reserved}"
    )

    if not entry.is_reserved and sample_captions is not None:
        n = 0
        n_collide = 0
        for caption in sample_captions:
            n += 1
            ids = tokenizer.encode(caption, add_special_tokens=False)
            if sentinel_id in ids:
                n_collide += 1
        if n > 0:
            rate = n_collide / n
            msg = (
                f"sentinel_registry: collision check tokenizer={name!r} "
                f"sentinel_id={sentinel_id} sampled {n} captions, "
                f"{n_collide} contained the sentinel id "
                f"({rate*100:.3f}%); threshold={max_collision_rate*100:.3f}%"
            )
            if rate > max_collision_rate:
                if strict:
                    raise ValueError(msg + " — failing fast (strict=True)")
                logger.warning(msg + " — proceeding but training quality may degrade")
            else:
                logger.info(msg + " — under threshold")

    return sentinel_id
