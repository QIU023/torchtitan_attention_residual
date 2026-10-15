# Local probe, never committed: each process prints, once per tower instance, whether dynamic CP has sub-CP groups.
import os

if os.environ.get("K3C_PROBE") == "1":
    try:
        from torchtitan.models.kimi_k3.vision_cp import encoder as _enc

        _orig = _enc.MoonViTCPEncoder.forward
        _seen = set()

        def _forward(self, *args, **kwargs):
            if id(self) not in _seen:
                _seen.add(id(self))
                print(f"DYNCP_PROBE rank={os.environ.get('RANK')} tower={id(self) % 10000} subgroups={len(self._cp_subgroups)} min_patches={self.dynamic_cp_min_patches}", flush=True)
            return _orig(self, *args, **kwargs)

        _enc.MoonViTCPEncoder.forward = _forward
    except Exception as exc:
        print(f"DYNCP_PROBE install failed: {exc}", flush=True)
