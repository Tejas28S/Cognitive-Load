import markdown
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
md_path = ROOT / "PROJECT_DOCUMENTATION.md"
html_path = ROOT / "_render_doc.html"  # temp, at repo root so relative image paths resolve

text = md_path.read_text(encoding="utf-8")

html_body = markdown.markdown(
    text,
    extensions=["extra", "toc", "sane_lists"],
    extension_configs={"toc": {"anchorlink": False}},
)

# Insert a page-break marker before each H2 section (major sections),
# except the very first one, so sections start on a new page in the PDF.
h2_positions = [m.start() for m in re.finditer(r"<h2", html_body)]
if h2_positions:
    offset = 0
    for i, pos in enumerate(h2_positions):
        if i == 0:
            continue
        marker = '<div class="page-break"></div>'
        insert_at = pos + offset
        html_body = html_body[:insert_at] + marker + html_body[insert_at:]
        offset += len(marker)

title = "Cognitive Load Estimation from EEG + fNIRS — Project Documentation"

CSS = """
@page {
  size: A4;
  margin: 20mm 18mm 20mm 18mm;
  @bottom-center {
    content: counter(page);
  }
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, "Segoe UI", "Inter", Arial, sans-serif;
  color: #1a1a2e;
  line-height: 1.55;
  font-size: 10.6pt;
  max-width: 100%;
}
h1, h2, h3, h4 { font-family: Georgia, "Times New Roman", serif; color: #111827; }
h1 {
  font-size: 24pt;
  margin: 0 0 4pt 0;
  border-bottom: 3px solid #4338ca;
  padding-bottom: 10pt;
}
.subtitle { color: #555; font-size: 10pt; margin: 0 0 18pt 0; font-family: Arial, sans-serif; }
h2 {
  font-size: 16pt;
  margin-top: 4pt;
  margin-bottom: 10pt;
  padding-bottom: 4pt;
  border-bottom: 1.5px solid #c7d2fe;
  color: #3730a3;
}
h3 {
  font-size: 12.5pt;
  color: #4338ca;
  margin-top: 16pt;
  margin-bottom: 6pt;
}
h4 { font-size: 11pt; color: #374151; margin-top: 12pt; }
p { margin: 6pt 0; text-align: left; }
ul, ol { margin: 6pt 0; padding-left: 20pt; }
li { margin: 3pt 0; }
code {
  background: #f1f0fb;
  color: #4338ca;
  padding: 1px 5px;
  border-radius: 3px;
  font-family: "Consolas", "IBM Plex Mono", monospace;
  font-size: 9.3pt;
}
pre {
  background: #1e1e2e;
  color: #e2e2f0;
  padding: 10pt 12pt;
  border-radius: 6px;
  font-size: 8.3pt;
  line-height: 1.4;
  page-break-inside: avoid;
  white-space: pre-wrap;
  word-break: break-word;
}
pre code { background: none; color: inherit; padding: 0; }
table {
  border-collapse: collapse;
  width: 100%;
  margin: 10pt 0;
  font-size: 9.3pt;
  page-break-inside: avoid;
}
th, td {
  border: 1px solid #d1d5db;
  padding: 5pt 7pt;
  text-align: left;
  vertical-align: top;
}
th { background: #eef2ff; color: #3730a3; font-weight: 600; }
tr:nth-child(even) td { background: #fafafa; }
img {
  max-width: 100%;
  max-height: 235mm;
  width: auto;
  height: auto;
  display: block;
  margin: 14pt auto;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
  page-break-inside: avoid;
  page-break-before: auto;
}
blockquote {
  border-left: 3px solid #4338ca;
  margin: 8pt 0;
  padding: 2pt 12pt;
  color: #4b5563;
  background: #f8f8fc;
}
hr { border: none; border-top: 1px solid #e5e7eb; margin: 18pt 0; }
strong { color: #111827; }
.page-break { page-break-before: always; break-before: page; }
.cover {
  text-align: center;
  padding-top: 22%;
  page-break-after: always;
  break-after: page;
}
.cover h1 { border: none; font-size: 30pt; margin-bottom: 10pt; }
.cover .tagline { font-size: 13pt; color: #4b5563; font-family: Arial, sans-serif; margin-bottom: 30pt; }
.cover .meta { font-size: 10pt; color: #6b7280; font-family: "Consolas", monospace; }
"""

full_html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>
<div class="cover">
  <h1>Cognitive Load Estimation<br/>from EEG + fNIRS</h1>
  <div class="tagline">Project Documentation — Architecture, Methodology, Results, and Status</div>
  <div class="meta">Generated 2026-08-30 &middot; branch <code>updated-frontend</code> &middot; commit 4c8c033</div>
</div>
{html_body}
</body>
</html>
"""

html_path.write_text(full_html, encoding="utf-8")
print(f"Wrote {html_path} ({len(full_html)} bytes)")
