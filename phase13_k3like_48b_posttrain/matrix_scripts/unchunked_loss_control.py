"""Run any kimi_k3 flavor with its chunked-loss wrapper replaced by plain CE.

The control the 30-layer DEP divergence needs. `kimi_k3_debugmodel_report_arch_pp8vp4`
pins `ChunkedLossWrapper(num_chunks=8)` in the flavor itself and torchtitan exposes no
CLI override for `loss`, so "same flavor, unchunked" is not expressible on the command
line. Patching the registry function in this process keeps the flavor definition
untouched -- nothing about the run changes except the loss object.

Two things follow automatically from the swap and are the reason it is a real control:
the trainer only sets `_skip_lm_head` when the loss is a ChunkedLossWrapper, and
`global_vocab_size` is carried over from the wrapper's inner loss config, so the plain
arm computes the same objective over the same vocabulary.

Usage: torchrun ... unchunked_loss_control.py --module kimi_k3 --config <flavor> ...
"""

from __future__ import annotations

import runpy
import sys

from torchtitan.components.loss import ChunkedLossWrapper


def _flavor_from_argv() -> str:
    for i, arg in enumerate(sys.argv):
        if arg == "--config":
            return sys.argv[i + 1]
        if arg.startswith("--config="):
            return arg.split("=", 1)[1]
    raise SystemExit("--config <flavor> is required")


def main() -> None:
    from torchtitan.models.kimi_k3 import config_registry

    flavor = _flavor_from_argv()
    original = getattr(config_registry, flavor)

    def unchunked() -> object:
        cfg = original()
        if not isinstance(cfg.loss, ChunkedLossWrapper.Config):
            raise SystemExit(
                f"{flavor} does not use ChunkedLossWrapper, so there is nothing to "
                f"control for (loss is {type(cfg.loss).__qualname__})"
            )
        cfg.loss = cfg.loss.loss_fn
        return cfg

    setattr(config_registry, flavor, unchunked)
    runpy.run_module("torchtitan.train", run_name="__main__")


if __name__ == "__main__":
    main()
