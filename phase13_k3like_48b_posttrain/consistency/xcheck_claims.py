"""同一量在树里出现多个不同值 -> 报冲突。代码注释与 logbook 文档一起扫。"""
import re, pathlib, collections

# (量名, 抓取正则) —— 只抓带单位/上下文的断言,避免抓到无关数字
CLAIMS = [
 ("DEP forward placed/total",   r"(\d+)\s*/\s*(\d+)\s+planned encode"),
 ("DEP idle slots",             r"(\d+)\s+idle slot"),
 ("DEP exhausted",              r"(\d+)\s+exhausted"),
 ("DEP upfront",                r"(\d+)\s+upfront"),
 ("DEP backward ran",           r"(\d+)\s+ran at a planned slot"),
 ("DEP tps delta",              r"([-+]?\d+\.\d+)%\)?\s*$|\(\*\*([-+]?\d+\.\d+)%\*\*\)"),
 ("DEP cost ratio",             r"cost ratio (?:of )?([0-9]+\.[0-9]+)"),
 ("DEP mem requirement GiB",    r"(?:>=|about|roughly|约)\s*(\d+)\s*GiB per GPU"),
 ("gate cells passed",          r"(\d+)\s*(?:of|/)\s*58"),
 ("ViT params",                 r"(\d+\.?\d*)\s*M\b.{0,24}(?:tower|ViT|MoonViT)"),
 ("ViT share of text",          r"(\d+\.?\d*)\s*(?:x|×)\s+the model it serves|(\d+\.\d+)%\s+of\s+(?:activated|the text)"),
 ("activated params",           r"(\d+\.\d+)\s*B\s+activated"),
 ("parallelize.py lines",       r"parallelize\.py[^.]{0,20}?(\d{3,4})\s*(?:行|lines)"),
]
roots = ["torchtitan/torchtitan/models/kimi_k3", "phase13_k3like_48b_posttrain", "Raising_PRs"]
seen = collections.defaultdict(lambda: collections.defaultdict(list))
for root in roots:
    for f in pathlib.Path(root).rglob("*"):
        if f.suffix not in (".py",".md") or "__pycache__" in str(f): continue
        try: txt = f.read_text(errors="ignore")
        except Exception: continue
        for name, pat in CLAIMS:
            for m in re.finditer(pat, txt, re.M):
                val = next((g for g in m.groups() if g), None)
                if val is None: continue
                ln = txt[:m.start()].count("\n")+1
                seen[name][val].append(f"{f.name}:{ln}")
for name in sorted(seen):
    vals = seen[name]
    if len(vals) < 2: continue
    print(f"\n## {name} -- {len(vals)} 个不同值")
    for v, locs in sorted(vals.items(), key=lambda kv: -len(kv[1])):
        print(f"   {v:<10} x{len(locs):<3} {', '.join(sorted(set(locs))[:4])}")
