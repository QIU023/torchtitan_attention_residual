"""Write remap_table.py's result into assign.py and the new head into splitlib.py: python apply_remap.py <new head>."""
import re
import sys

import assign as A
import remap_table as R

NEW = sys.argv[1]
s = open("assign.py").read()
# 1. the engine file's ranges
start = s.index("    IMPL: [\n")
end = s.index("    ],\n", start)
body = "".join(f"    (({lo},{hi}), \"{name}\"),\n" for (lo, hi), name in R.RANGES)
s = s[:start] + "    IMPL: [\n\n" + body + s[end:]
# 2. the variant keys, one pass so a key moving onto another old key cannot be mapped twice
vstart = s.index("VARIANTS = {\n    IMPL: {")
vend = s.index('    "verl/workers/engine/torchtitan/utils.py": {},', vstart)
block = s[vstart:vend]
keys = R.VARIANT_KEYS
missing = set(A.VARIANTS[A.IMPL]) - set(keys)
assert not missing, f"variant keys without a new index: {sorted(missing)}"
block = re.sub(r"(?<![0-9])(\d+): \[", lambda m: f"{keys[int(m.group(1))]}: [", block)
s = s[:vstart] + block + s[vend:]
# 3. the moves
s = re.sub(r"(?m)^    IMPL: \[\(\(.*\n", "    IMPL: " + repr(R.MOVES) + ",\n", s, count=1)
open("assign.py", "w").write(s)
t = open("splitlib.py").read()
t = re.sub(r'HEAD = "[0-9a-f]+"', f'HEAD = "{NEW}"', t, count=1)
open("splitlib.py", "w").write(t)
print("assign.py and splitlib.py rewritten for", NEW)
