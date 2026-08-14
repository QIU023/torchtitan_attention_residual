"""List every declaration conflict the DRIVER hits, in one run.

probe_declaration_conflicts reads after apply_tp, so it only sees modules the
imperative plan already touched. The driver meets a different set -- the parents
the plan covered via their children -- and those conflicts only surface one per
crash. This catches them at _distribute_states and keeps going.
"""
import os, sys
if os.environ.get("DCF"):
    import importlib.abc, importlib.machinery
    class T(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "torchtitan.protocols.module": return None
            sys.meta_path.remove(self)
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            oe = spec.loader.exec_module
            def ex(mod):
                oe(mod)
                cls = mod.Module
                orig = cls._distribute_states
                hits = []
                def probe(self, *a, **k):
                    try:
                        return orig(self, *a, **k)
                    except ValueError as e:
                        if "already a DTensor" in str(e):
                            msg = str(e).split(". This usually")[0]
                            if msg not in hits:
                                hits.append(msg)
                                print(f"[dcf] {type(self).__name__}: {msg}",
                                      file=sys.stderr, flush=True)
                            return None
                        raise
                cls._distribute_states = probe
            spec.loader.exec_module = ex
            return spec
    sys.meta_path.insert(0, T())
