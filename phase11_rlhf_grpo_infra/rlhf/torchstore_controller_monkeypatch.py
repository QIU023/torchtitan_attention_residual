"""Promote 5 sync @endpoint methods on torchstore Controller to async.

Monarch's actor_mesh requires all @endpoint methods on a class to be
*consistently* async or sync. torchstore 0.1.2 mixes 2 async (init,
teardown) with 5 sync (get_controller_strategy, locate_volumes,
notify_put, keys, notify_delete). Without this patch the Controller
fails at actor instantiation with:

    ValueError: <class 'torchstore.controller.Controller'> mixes both
    async and sync endpoints.

This module must be imported in EVERY process that touches torchstore
Controller — both the main GRPO process and each monarch-spawned worker
actor. The workers don't import the main script's module-level code, so
this patch must be applied in their bootstrap import chain too.

Idempotent: re-importing is safe (the patch checks if the endpoint is
already async before wrapping).
"""
import asyncio
import warnings


def _apply():
    try:
        from torchstore.controller import Controller
        from monarch.actor import endpoint
    except Exception as e:
        warnings.warn(f"torchstore_controller_monkeypatch: import failed ({e})")
        return False
    sync_names = (
        "get_controller_strategy",
        "locate_volumes",
        "notify_put",
        "keys",
        "notify_delete",
    )
    n = 0
    for name in sync_names:
        ep = Controller.__dict__.get(name)
        if ep is None:
            continue
        src = getattr(ep, "_method", None) or getattr(ep, "_func", None)
        if src is None:
            continue
        if asyncio.iscoroutinefunction(src):
            continue  # already async (re-import safe)

        def _make_async(fn):
            async def _async_wrapper(self, *a, **kw):
                return fn(self, *a, **kw)
            _async_wrapper.__name__ = fn.__name__
            _async_wrapper.__qualname__ = getattr(fn, "__qualname__", fn.__name__)
            return _async_wrapper

        setattr(Controller, name, endpoint(_make_async(src)))
        n += 1
    return n > 0


_applied = _apply()
