# Local harness only: give every torchrun rank its own Triton cache, so ranks of one run
# never compile the same kernel into the same directory at once.
import os

_base = os.environ.get("TRITON_CACHE_BASE")
_rank = os.environ.get("LOCAL_RANK")
if _base and _rank is not None:
    os.environ["TRITON_CACHE_DIR"] = os.path.join(_base, f"rank{_rank}")
