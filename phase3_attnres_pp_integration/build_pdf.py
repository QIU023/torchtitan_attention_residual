"""Render PP_Adapter_Flow.md -> dark-themed HTML (inlined SVG) -> PDF.

Standalone build helper for the phase-3 adapter doc. Uses the `markdown`
package for MD->HTML and headless Chrome for HTML->PDF so no LaTeX /
pandoc / wkhtmltopdf install is required.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
MD = HERE / "PP_Adapter_Flow.md"
SVG = HERE / "pp_adapter_flow_dark.svg"
HTML = HERE / "PP_Adapter_Flow.html"
PDF = HERE / "PP_Adapter_Flow.pdf"

CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"

import markdown  # noqa: E402

body = markdown.markdown(
    MD.read_text(encoding="utf-8"),
    extensions=["extra", "tables", "fenced_code", "codehilite", "sane_lists"],
)

# Inline the SVG in place of the <img> the markdown renderer emitted, so
# the headless print step never needs file access to a sibling asset.
svg_inline = SVG.read_text(encoding="utf-8")
svg_inline = re.sub(r"<\?xml.*?\?>", "", svg_inline, flags=re.DOTALL).strip()
body = re.sub(
    r'<p>\s*<img[^>]*pp_adapter_flow_dark\.svg[^>]*>\s*</p>',
    f'<div class="diagram">{svg_inline}</div>',
    body,
)

CSS = """
:root { color-scheme: dark; }
* { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
html, body { background: #000000; margin: 0; }
body {
  color: #c9d1d9;
  font-family: "Segoe UI", "Microsoft YaHei", Helvetica, Arial, sans-serif;
  font-size: 13px; line-height: 1.7;
  padding: 36px 48px;
}
h1, h2, h3 { color: #f0f6fc; font-weight: 700; line-height: 1.3; }
h1 { font-size: 26px; border-bottom: 2px solid #30363d; padding-bottom: 12px; }
h2 { font-size: 20px; margin-top: 34px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
h3 { font-size: 15px; margin-top: 22px; color: #79b8ff; }
a { color: #58a6ff; text-decoration: none; }
strong { color: #f0f6fc; }
blockquote {
  border-left: 4px solid #e3b341; background: #161b22;
  margin: 14px 0; padding: 10px 16px; color: #d4af5a; border-radius: 0 6px 6px 0;
}
code {
  background: #161b22; color: #ff9492;
  padding: 1.5px 5px; border-radius: 4px;
  font-family: "Cascadia Code", "Consolas", monospace; font-size: 12px;
}
pre { background: #0d1117; border: 1px solid #30363d; border-radius: 8px;
  padding: 14px 16px; overflow-x: auto; }
pre code { background: none; color: #c9d1d9; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 16px 0; font-size: 12px; }
th, td { border: 1px solid #30363d; padding: 7px 11px; text-align: left; vertical-align: top; }
th { background: #161b22; color: #f0f6fc; }
tr:nth-child(even) td { background: #0d1117; }
hr { border: none; border-top: 1px solid #21262d; margin: 28px 0; }
.diagram {
  text-align: center; margin: 0;
  page-break-before: always; page-break-after: always; page-break-inside: avoid;
  height: 178mm;                /* printable height on A4 landscape, 14mm margins */
  display: flex; align-items: center; justify-content: center;
}
.diagram svg {
  height: 100%; width: auto;    /* height is the binding constraint (ratio 1.31) */
  max-width: 100%;
  border: 1px solid #30363d; border-radius: 10px;
}
@page { size: A4 landscape; margin: 14mm; background: #000000; }
"""

HTML.write_text(
    f"<!doctype html><html><head><meta charset='utf-8'>"
    f"<style>{CSS}</style></head><body>{body}</body></html>",
    encoding="utf-8",
)
print(f"wrote {HTML}")

subprocess.run(
    [
        CHROME, "--headless=new", "--disable-gpu", "--no-sandbox",
        "--no-pdf-header-footer",
        f"--print-to-pdf={PDF}",
        HTML.as_uri(),
    ],
    check=True,
)
print(f"wrote {PDF}")
