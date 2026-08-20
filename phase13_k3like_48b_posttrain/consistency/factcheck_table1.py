"""树里出现的 K3 真实架构数字 vs 报告 Table 1。只报可疑,不改。"""
import re, pathlib
T1 = {"layers":"93","total":"2.78T","activated":"104.2B","hidden":"7168",
      "routed_experts":"896","active_per_token":"16","shared":"2","attn_heads":"96",
      "composition":"69 KDA + 24 MLA","vit_total":"401M","vit_layers":"27",
      "vit_patch":"14","vit_heads":"12","ctx":"1M","moe_hidden":"3072","latent":"3584"}
# 形如"报告说 X"或对真实规模的断言里,出现与 Table 1 冲突的数
SUSPECT = [
 (r"\b(61|69|93)\s*(?:层|layers?)\b",           "层数:K3=93,注意力构成 69 KDA + 24 MLA"),
 (r"\b(384|896)\s*(?:routed )?experts?\b",       "routed experts:896"),
 (r"\b(8|16)\s*(?:experts? )?active",            "active per token:16"),
 (r"\b(32\.6|104\.2|105\.8)\s*B\b",              "activated:104.2B(我们算 105.8B,差 1.5%)"),
 (r"\b(1\.04|2\.78|2\.8)\s*T\b",                 "total:2.78T"),
 (r"\b(64|96)\s*(?:attention )?heads\b",         "attn heads:96"),
 (r"\b(2048|3072)\b.*expert",                    "MoE hidden per expert:3072"),
 (r"\b(128K|1M)\s*(?:token|context)",            "context:1M"),
 (r"\b(12|16)\s*(?:ViT )?heads?.*ViT|ViT.*\b(12|16)\s*heads?", "ViT heads:12"),
]
roots = ["torchtitan/torchtitan/models/kimi_k3", "phase13_k3like_48b_posttrain", "Raising_PRs"]
out = {}
for root in roots:
    for f in pathlib.Path(root).rglob("*"):
        if f.suffix not in (".py",".md") or "__pycache__" in str(f): continue
        try: txt=f.read_text(errors="ignore")
        except Exception: continue
        for pat, note in SUSPECT:
            for m in re.finditer(pat, txt, re.I):
                ln = txt[:m.start()].count("\n")+1
                line = txt.split("\n")[ln-1].strip()[:96]
                out.setdefault(note, []).append(f"{f}:{ln}  {line}")
for note, rows in out.items():
    print(f"\n## {note}  ({len(rows)})")
    for r in sorted(set(rows))[:6]: print("   ", r)
