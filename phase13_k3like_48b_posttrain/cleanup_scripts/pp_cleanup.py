"""Cleanup pass for k3_pp_text's pipeline_adapter.py. Run with the file path."""

import re
import sys
import pathlib

p = pathlib.Path(sys.argv[1])
s = p.read_text()

# 1. The env-gated debug printer. knobs.py in this same branch argues that
#    upstream will not take env-var control; this is the other one.
s = re.sub(
    r'\n\n_DBG = os\.environ\.get\("ATTNRES_ADAPTER_DBG"\) == "1"\n\n\n'
    r'def _dbg\(msg: str\) -> None:\n    if _DBG:\n'
    r'        rank = os\.environ\.get\("RANK", "\?"\)\n'
    r'        print\(f"\[adapter-dbg rank=\{rank\}\] \{msg\}", flush=True\)\n',
    "\n",
    s,
)
s = re.sub(r"\n[ \t]*_dbg\((?:[^()]|\([^()]*\))*\)\n", "\n", s)
s = re.sub(r"\n[ \t]*_dbg\(\n(?:[^)]*\n)*?[ \t]*\)\n", "\n", s)
s = re.sub(r"\nimport os\n", "\n", s, count=1)

# 2. ASCII only in comments and docstrings we wrote.
for a, b in (
    ("§", "section "),
    ("→", "->"),
    ("—", "--"),
    ("‘", "'"),
    ("’", "'"),
    ("“", '"'),
    ("”", '"'),
):
    s = s.replace(a, b)

# 3. Bold markdown structure in docstrings; torchtitan's do not use it.
s = re.sub(r"\*\s\*\*(Different rank|Same rank)\*\*", lambda m: "* " + m.group(1), s)

p.write_text(s)

bad = [i + 1 for i, line in enumerate(s.splitlines()) if any(ord(c) > 127 for c in line)]
print(f"non-ascii lines left: {bad}")
print(f"_dbg refs left: {s.count('_dbg')}  _DBG refs left: {s.count('_DBG')}")
print(f"'import os' left: {'import os' in s}")
