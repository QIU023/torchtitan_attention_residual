"""LOCAL RUN HACK (not committed): let the Attention Gym KDA kernels run on Hopper.

The branch's guard is main's -- `capability not in {(10, 0), (10, 3)}` -- so an H100 (SM 9.0)
is refused before any kernel runs. The capability widening we carried was reverted on
Shuhua's review (it is unrelated to tensor parallelism), so a Hopper box needs this locally.
Apply to a throwaway copy of the tree, never commit it.

    python hacks/kda_capability_hack.py /path/to/tree
"""

import pathlib
import sys

root = pathlib.Path(sys.argv[1])
p = root / "torchtitan/models/kimi_k3/kda.py"
s = p.read_text()

old = """        if capability not in {(10, 0), (10, 3)}:
            raise RuntimeError(
                "Attention Gym KDA requires Blackwell SM100/SM103; "
                f"got CUDA capability {capability}."
            )
"""
assert s.count(old) == 1, f"guard not found as expected (count={s.count(old)})"
new = """        if capability < (8, 0):  # LOCAL RUN HACK (not committed)
            raise RuntimeError(
                "Attention Gym KDA requires CUDA capability 8.0 or newer; "
                f"got {capability}."
            )
"""
p.write_text(s.replace(old, new))
print("kda.py: capability guard widened to >= 8.0 in", root)
