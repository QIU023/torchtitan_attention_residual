"""Print the tensor KIND at every DTensor boundary crossing, in a real trainer run.

Written for the vision tower's dynamic-CP backward failure on
``tp_declarative_refactor``:

    RuntimeError: the local_tensor argument only accepts torch.Tensor but got DTensor

``--debug.detect-anomaly`` places the forward in ``_encode_images_dynamic_cp``, but not
WHICH of its conversions, nor what the operand actually was. An earlier attempt to answer
that by rebuilding ``ParallelDims`` outside the trainer died with "Unknown c10d backend
type FAKE" -- the construction needs the same backend the trainer uses. So instrument the
real run instead: drop this file's directory on ``PYTHONPATH`` as ``sitecustomize.py`` and
every rank patches itself at interpreter startup, before torchtitan is imported.

    cp probe.py /tmp/kindprobe/sitecustomize.py
    PYTHONPATH=/tmp/kindprobe:$TITAN KIND_PROBE=1 torchrun ... -m torchtitan.train ...

Set KIND_PROBE_ALL=1 to log every crossing (loud). The default logs only crossings whose
operand is the wrong kind, which is the failure itself, plus a two-frame caller location.
"""
import os
import sys
import traceback

if os.environ.get("KIND_PROBE"):
    import torch
    from torch.distributed.tensor import DTensor

    _RANK = os.environ.get("RANK", "?")
    _ALL = bool(os.environ.get("KIND_PROBE_ALL"))
    _seen = set()

    def _where(skip=2):
        frames = traceback.extract_stack()[:-skip]
        return " <- ".join(f"{f.filename.split('/')[-1]}:{f.lineno}" for f in frames[-2:])

    def _kind(t):
        if isinstance(t, DTensor):
            return f"DTensor{tuple(str(p) for p in t.placements)}"
        if isinstance(t, torch.Tensor):
            return "plain"
        return type(t).__name__

    def _log(tag, msg, dedup_key):
        # One line per distinct call site: a failing conversion inside a per-image loop
        # would otherwise print once per pass per layer and bury the first occurrence.
        if dedup_key in _seen:
            return
        _seen.add(dedup_key)
        print(f"[kind rank{_RANK}] {tag} {msg}", file=sys.stderr, flush=True)

    _orig_from_local = DTensor.from_local

    def _probe_from_local(local_tensor, *args, **kwargs):
        site = _where()
        bad = isinstance(local_tensor, DTensor)
        if bad or _ALL:
            _log(
                "BAD " if bad else "ok  ",
                f"from_local(local={_kind(local_tensor)}) at {site}",
                ("fl", site, bad),
            )
        return _orig_from_local(local_tensor, *args, **kwargs)

    DTensor.from_local = staticmethod(_probe_from_local)

    _orig_to_local = DTensor.to_local

    def _probe_to_local(self, *args, **kwargs):
        if _ALL:
            _log("ok  ", f"to_local(self={_kind(self)}) at {_where()}", ("tl", _where()))
        return _orig_to_local(self, *args, **kwargs)

    DTensor.to_local = _probe_to_local

    # The failure is in BACKWARD, so it may be raised from inside an autograd node rather
    # than through the wrapper above. When that happens, dump the KIND of every tensor
    # local in each frame of the traceback -- that is the information the bare
    # RuntimeError withholds and the reason this failure took five 8-GPU runs to not
    # localise.
    _orig_excepthook = sys.excepthook

    def _kind_excepthook(exc_type, exc, tb):
        print(f"[kind rank{_RANK}] === tensor kinds per frame ===", file=sys.stderr)
        walk = tb
        while walk is not None:
            frame = walk.tb_frame
            tensors = {
                k: _kind(v)
                for k, v in frame.f_locals.items()
                if isinstance(v, torch.Tensor)
            }
            if tensors:
                loc = f"{frame.f_code.co_filename.split('/')[-1]}:{walk.tb_lineno}"
                print(f"[kind rank{_RANK}] {loc} {tensors}", file=sys.stderr)
            walk = walk.tb_next
        sys.stderr.flush()
        _orig_excepthook(exc_type, exc, tb)

    sys.excepthook = _kind_excepthook

    print(f"[kind rank{_RANK}] probe installed (all={_ALL})", file=sys.stderr, flush=True)
