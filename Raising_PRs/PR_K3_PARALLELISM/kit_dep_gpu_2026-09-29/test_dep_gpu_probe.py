"""Local probe (never committed): the DEP NCCL GPU test's bubble case, printing each rank's
exception before the harness tears the process group down."""

import sys
import traceback
import unittest

import torch
from torch.testing._internal.distributed._tensor.common_dtensor import DTensorTestBase, with_comms

from tests.unit_tests.cpu.test_kimi_k3_vision_dep import _VisionDepChecks


class TestProbe(_VisionDepChecks, DTensorTestBase):
    gelu = True
    exact = False

    @property
    def device_type(self) -> str:
        return "cuda"

    @property
    def world_size(self) -> int:
        return 4

    @with_comms
    def test_bubble(self):
        try:
            self._check(bubble=True, frozen_tower=False)
            print(f"[probe rank {self.rank}] passed", file=sys.stderr, flush=True)
        except BaseException:
            print(f"[probe rank {self.rank}] FAILED:\n{traceback.format_exc()}", file=sys.stderr, flush=True)
            raise


if __name__ == "__main__":
    unittest.main()
