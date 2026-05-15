"""Tests for the tokenizer-aware sentinel registry (phase 6 task B4).

Coverage:
* registry hit on Llama-3.1 family
* fallback for unknown tokenizer
* collision check warns under threshold, raises in strict mode
* unknown role is rejected
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Iterable

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKSPACE = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _WORKSPACE)

from phase5_vlm_multimodal_sft.sentinel_registry import (  # noqa: E402
    SENTINEL_REGISTRY,
    SentinelEntry,
    resolve_sentinel,
)


@dataclass
class _StubTokenizer:
    """Minimal tokenizer stub that mimics HuggingFace's interface."""
    name_or_path: str
    sentinel_in_first_n: int = 0  # encode returns sentinel for the first N captions

    def __post_init__(self):
        self._calls = 0

    def encode(self, text: str, add_special_tokens: bool = False):
        self._calls += 1
        ids = [ord(c) % 31_000 + 100 for c in text[:8]]  # arbitrary but deterministic
        if self._calls <= self.sentinel_in_first_n:
            ids.append(32_000)  # collision
        return ids


def test_registry_hit_llama31():
    tok = _StubTokenizer(name_or_path="NousResearch/Meta-Llama-3.1-8B")
    sid = resolve_sentinel(tok)
    assert sid == 32_000
    assert SENTINEL_REGISTRY["llama-3.1"].decoded_str == "utility"


def test_registry_hit_llama3():
    tok = _StubTokenizer(name_or_path="meta-llama/Llama-3-8b-hf")
    # 'llama-3' substring matches both 'llama-3' and 'llama-3.1' entries;
    # the first one matched in dict-iteration order wins. With Python 3.7+
    # dict order = insertion order. The test should be robust to either —
    # both entries map to id 32000 anyway.
    sid = resolve_sentinel(tok)
    assert sid == 32_000


def test_registry_hit_kimi():
    tok = _StubTokenizer(name_or_path="moonshotai/Kimi-Linear-48B-A3B-Base")
    sid = resolve_sentinel(tok)
    assert sid == 163_839
    assert SENTINEL_REGISTRY["kimi"].is_reserved is True


def test_unknown_tokenizer_falls_back():
    tok = _StubTokenizer(name_or_path="some-org/MysteryModel-7B")
    sid = resolve_sentinel(tok)
    # Falls back to id 32000 (preserves the original hardcoded behavior).
    assert sid == 32_000


def test_collision_check_warns_under_threshold(caplog):
    import logging
    caplog.set_level(logging.INFO)
    tok = _StubTokenizer(
        name_or_path="NousResearch/Meta-Llama-3.1-8B",
        sentinel_in_first_n=0,  # zero collisions
    )
    sid = resolve_sentinel(
        tok,
        sample_captions=[f"caption-{i}" for i in range(100)],
        max_collision_rate=0.01,
    )
    assert sid == 32_000
    # Info-level log says "under threshold"
    assert any("under threshold" in r.message for r in caplog.records)


def test_collision_check_warns_over_threshold(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    tok = _StubTokenizer(
        name_or_path="NousResearch/Meta-Llama-3.1-8B",
        sentinel_in_first_n=10,  # 10/100 = 10% collision rate
    )
    sid = resolve_sentinel(
        tok,
        sample_captions=[f"caption-{i}" for i in range(100)],
        max_collision_rate=0.001,  # 0.1% threshold
    )
    assert sid == 32_000  # still returns id even on warning
    assert any(
        r.levelno == logging.WARNING and "training quality may degrade" in r.message
        for r in caplog.records
    )


def test_collision_check_strict_raises():
    tok = _StubTokenizer(
        name_or_path="NousResearch/Meta-Llama-3.1-8B",
        sentinel_in_first_n=10,
    )
    with pytest.raises(ValueError, match="failing fast"):
        resolve_sentinel(
            tok,
            sample_captions=[f"caption-{i}" for i in range(100)],
            max_collision_rate=0.001,
            strict=True,
        )


def test_reserved_tokenizer_skips_collision_check():
    """When sentinel is_reserved=True (e.g. Kimi), no collision check runs
    even when sample_captions is provided — reserved ids by definition
    cannot appear in real text."""
    tok = _StubTokenizer(
        name_or_path="moonshotai/Kimi-Linear-48B",
        sentinel_in_first_n=100,  # would trigger over-threshold
    )
    # Should return without inspecting sample_captions
    sid = resolve_sentinel(
        tok,
        sample_captions=[f"caption-{i}" for i in range(100)],
        max_collision_rate=0.0,
        strict=True,
    )
    assert sid == 163_839


def test_unknown_role_rejected():
    tok = _StubTokenizer(name_or_path="NousResearch/Meta-Llama-3.1-8B")
    with pytest.raises(NotImplementedError, match="audio"):
        resolve_sentinel(tok, role="audio")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
