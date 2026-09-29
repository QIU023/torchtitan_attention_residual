"""Local probe (never committed): the DEP NCCL GPU test's pipeline against its single-device
reference, printing the max abs / rel difference of every checked tensor per step and rank."""

import sys
import unittest

import torch
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms

from tests.unit_tests.cpu.test_kimi_k3_vision_dep import _run_single_device, _VisionDepChecks, STEPS


def _diff(a, b):
    a, b = a.double(), b.double()
    abs_d = (a - b).abs().max().item()
    rel_d = ((a - b).abs() / b.abs().clamp_min(1e-30)).max().item()
    return abs_d, rel_d


class TestProbe(_VisionDepChecks, DTensorTestBase):
    gelu = True
    exact = False
    lr = 1e-3

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def world_size(self) -> int:
        return 4

    def _report(self, bubble: bool) -> None:
        reference = _run_single_device(False, self._device(), self.gelu, self.lr)
        evals: list = []
        history, _ = self._run_pipeline(bubble=bubble, frozen_tower=False, evals=evals)
        lines = []
        for step in range(STEPS):
            grads, losses = history[step]
            ref_grads, ref_losses = reference[step]
            for name, grad in grads.items():
                if grad is None or ref_grads[name] is None:
                    continue
                a, r = _diff(grad, ref_grads[name])
                over = a > 1e-5 + 1.3e-6 * ref_grads[name].abs().max().item()
                lines.append(f"rank {self.rank} bubble={bubble} step {step} {name} numel {grad.numel()} abs {a:.2e} rel {r:.2e}{' OVER-DEFAULT' if over else ''}")
            if losses:
                for i, (x, y) in enumerate(zip(losses, ref_losses)):
                    a, r = _diff(x, y)
                    lines.append(f"rank {self.rank} bubble={bubble} step {step} loss[{i}] abs {a:.2e} rel {r:.2e}")
        print("\n".join(lines), file=sys.stderr, flush=True)

    @with_comms
    def test_k25(self):
        self._report(False)

    @with_comms
    def test_bubble(self):
        self._report(True)


if __name__ == "__main__":
    unittest.main()
