# Diagnostic only: name the FSDP parameter whose post_backward runs without an all-gather.
try:
    from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
    _orig = FSDPParam.to_accumulated_grad_if_needed
    def _guarded(self):
        if not hasattr(self, "_unsharded_param"):
            import sys
            print(f"[fsdp-diag] post_backward without all-gather: {getattr(self, '_param_fqn', '?')} "
                  f"sharded_state={getattr(self, 'sharded_state', '?')}", file=sys.stderr, flush=True)
            return
        return _orig(self)
    FSDPParam.to_accumulated_grad_if_needed = _guarded
except Exception as e:  # pragma: no cover
    print("[fsdp-diag] patch failed:", e)
