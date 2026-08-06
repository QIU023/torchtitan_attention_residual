"""Minimal nested ``kimi_k3`` config so transformers' AutoConfig can load a
multimodal export.

OURS, not a copy of the release. The release ships its own
``configuration_kimi_k3.py`` defining the multimodal config; we do not have that
file locally (only the one defining the TEXT config class). Consumers that go
through ``AutoConfig`` -- veRL's ``HFModelConfig`` is the one that matters here --
fail with "Transformers does not recognize this architecture" without something
registered for ``model_type: kimi_k3``.

Deliberately minimal: enough structure for a consumer to read dimensions and
walk into ``text_config`` / ``vision_config``. It is NOT a reimplementation of
the release's config semantics, and anything that needs those should use the
release file once it is available.

Shipped beside a multimodal export and referenced from its ``auto_map``.
"""

from __future__ import annotations

from transformers.configuration_utils import PretrainedConfig


class KimiK3TextConfig(PretrainedConfig):
    model_type = "kimi_linear"

    def __init__(self, **kwargs):
        # Every field is carried through as an attribute rather than enumerated:
        # the authority on which text fields exist is
        # hf_key_map.titan_config_to_official, and duplicating that list here
        # would give the two a way to disagree.
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__(**kwargs)


class KimiK3VisionConfig(PretrainedConfig):
    model_type = "kimi_k3_vision"

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        super().__init__(**kwargs)


class KimiK3Config(PretrainedConfig):
    model_type = "kimi_k3"
    sub_configs = {"text_config": KimiK3TextConfig, "vision_config": KimiK3VisionConfig}

    def __init__(
        self,
        text_config: dict | None = None,
        vision_config: dict | None = None,
        media_placeholder_token_id: int = 163605,
        **kwargs,
    ):
        self.text_config = KimiK3TextConfig(**(text_config or {}))
        self.vision_config = KimiK3VisionConfig(**(vision_config or {}))
        self.media_placeholder_token_id = media_placeholder_token_id
        super().__init__(**kwargs)

    # Read-only delegations, matching the release's shape: a flat top-level
    # hidden_size on a nested config is not just redundant, it collides with
    # these and raises "property has no setter" on assignment.
    @property
    def hidden_size(self) -> int:
        return self.text_config.hidden_size

    @property
    def vocab_size(self) -> int:
        return self.text_config.vocab_size

    @property
    def num_hidden_layers(self) -> int:
        return self.text_config.num_hidden_layers
