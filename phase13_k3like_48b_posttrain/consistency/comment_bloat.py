"""注释体量审计。两条仓库规矩:logbook CLAUDE.md 说 inline comment 一行为限、WHY 进文档;
torchtitan .claude/CLAUDE.md 说 minimize comments、code should be self-documenting。

    python3 consistency/comment_bloat.py [--top N]

只报告。>=6 行的连续 # 块和 >=12 行的 docstring 各自列出,按体量排序 —— 那是人类 reviewer
最先放弃阅读的地方。
"""
import ast, pathlib, sys
MOD = pathlib.Path("torchtitan/torchtitan/models/kimi_k3")
top = int(sys.argv[sys.argv.index("--top")+1]) if "--top" in sys.argv else 20
items = []
tot_code = tot_cmt = 0
for f in sorted(MOD.rglob("*.py")):
    if "__pycache__" in str(f): continue
    lines = f.read_text(errors="ignore").split("\n")
    tot_code += len(lines)
    cur = 0
    for i, l in enumerate(lines, 1):
        if l.lstrip().startswith("#"):
            cur += 1; tot_cmt += 1
        else:
            if cur >= 6: items.append((cur, f"{f.name}:{i-cur}", "# 块"))
            cur = 0
    try: t = ast.parse("\n".join(lines))
    except Exception: continue
    for n in ast.walk(t):
        if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef,ast.Module)):
            d = ast.get_docstring(n)
            if not d: continue
            nl = d.count("\n")+1; tot_cmt += nl
            if nl >= 12:
                items.append((nl, f"{f.name}:{getattr(n,'lineno',1)}", f"docstring {getattr(n,'name','<module>')}"))
print(f"模块 {tot_code} 行,注释/docstring {tot_cmt} 行 ({100*tot_cmt/tot_code:.0f}%)\n")
print(f"{'行数':>5}  {'位置':<34} 类型")
for n, loc, kind in sorted(items, reverse=True)[:top]:
    print(f"{n:>5}  {loc:<34} {kind}")
print(f"\n共 {len(items)} 处超标")
