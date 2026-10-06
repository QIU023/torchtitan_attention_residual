"""Logbook kit only: make the PP memory and transport switches announce themselves from each rank. Counts the
activation-storage backends' puts (host and remote) and records each pipeline stage's p2p_per_edge, written to
FEATURE_PROBE_LOG.rank<N> at exit."""

import atexit
import os

if os.environ.get("FEATURE_PROBE_LOG") and "RANK" in os.environ:  # the torchrun parent has no RANK
    from collections import Counter

    _counts: Counter = Counter()

    def _wrap_put(cls, name):
        orig = cls.put

        def put(self, tensor, stream):
            _counts[f"{name}_puts"] += 1
            _counts[f"{name}_bytes"] += tensor.numel() * tensor.element_size()
            return orig(self, tensor, stream)

        cls.put = put

    try:
        from torchtitan.distributed import activation_storage as _storage

        _wrap_put(_storage.HostBackend, "host")
        _wrap_put(_storage.RemoteBackend, "remote")
    except ImportError:
        pass

    from torch.distributed.pipelining import stage as _stage

    _orig_init = _stage._PipelineStageBase.__init__

    def _init(self, *args, **kwargs):
        _orig_init(self, *args, **kwargs)
        _counts[f"stage{self.stage_index}_p2p_per_edge={self.p2p_per_edge}"] += 1

    _stage._PipelineStageBase.__init__ = _init

    @atexit.register
    def _dump():
        with open(f"{os.environ['FEATURE_PROBE_LOG']}.rank{os.environ['RANK']}", "w") as f:
            for key, value in sorted(_counts.items()):
                f.write(f"{key} {value}\n")
