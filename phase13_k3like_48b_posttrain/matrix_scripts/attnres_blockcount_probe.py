"""How many blocks enter each AttnRes aggregation, naive path vs delta path?

The PP adapter's cache is a memory optimisation: with it on or off the model must
compute the same thing. Measured, it does not -- `pp2` on `report_arch` differs
at STEP 1 by 0.03, and at step 1 the pseudo-query is still zero-initialised, so
the aggregation is exactly the mean over its inputs. A mean that far off usually
means the COUNT is wrong, not the summation order.

This counts them. Drop the directory on PYTHONPATH as `sitecustomize.py` and every
rank patches itself at interpreter startup, before torchtitan is imported -- the
same trick the KIND probe uses, and for the same reason: it runs inside the real
trainer instead of trying to rebuild one.

    BLOCKCOUNT_PROBE=1 PYTHONPATH=/tmp/bcprobe:$TITAN torchrun ... -m torchtitan.train ...

Prints one line per distinct (layer call index, N) pair on rank 0, so a run with a
stable pattern stays short and a run whose count drifts shows every step.
"""

import os
import sys

if os.environ.get("BLOCKCOUNT_PROBE"):
    import torch

    _RANK = os.environ.get("RANK", "?")
    _seen = set()
    _calls = [0]

    def _install():
        from torchtitan.models.kimi_k3 import attn_res

        orig_tensor = attn_res.block_attn_res_tensor
        orig_list = attn_res.block_attn_res

        def probe_tensor(prefix_sum_BLD, block_residual_TND, proj, norm):
            # values = carrier columns + the current partial, so N+1 inputs.
            n = block_residual_TND.shape[1] + 1
            key = ("tensor", _calls[0], n)
            if key not in _seen:
                _seen.add(key)
                print(
                    f"[blockcount rank{_RANK}] call={_calls[0]:3d} inputs={n}",
                    file=sys.stderr,
                    flush=True,
                )
            _calls[0] += 1
            return orig_tensor(prefix_sum_BLD, block_residual_TND, proj, norm)

        def probe_list(blocks, partial_block, proj, norm):
            n = len(blocks) + 1
            key = ("list", _calls[0], n)
            if key not in _seen:
                _seen.add(key)
                print(
                    f"[blockcount rank{_RANK}] call={_calls[0]:3d} inputs={n}",
                    file=sys.stderr,
                    flush=True,
                )
            _calls[0] += 1
            return orig_list(blocks, partial_block, proj, norm)

        attn_res.block_attn_res_tensor = probe_tensor
        attn_res.block_attn_res = probe_list
        # The model module imported the names directly, so patch there too.
        try:
            from torchtitan.models.kimi_k3 import attn_res_model

            attn_res_model.block_attn_res_tensor = probe_tensor
            attn_res_model.block_attn_res = probe_list
        except ImportError:
            pass
        print(f"[blockcount rank{_RANK}] probe installed", file=sys.stderr, flush=True)

    # torchtitan is not importable at interpreter startup, so defer until the
    # trainer has imported it. A meta_path hook fires exactly once, on the import
    # that matters, instead of guessing at a delay.
    import importlib.abc
    import importlib.machinery

    class _Trigger(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname == "torchtitan.models.kimi_k3.attn_res_model":
                sys.meta_path.remove(self)
                spec = importlib.machinery.PathFinder.find_spec(fullname, path)
                if spec is not None:
                    orig_exec = spec.loader.exec_module

                    def exec_module(module):
                        orig_exec(module)
                        _install()

                    spec.loader.exec_module = exec_module
                return spec
            return None

    sys.meta_path.insert(0, _Trigger())
