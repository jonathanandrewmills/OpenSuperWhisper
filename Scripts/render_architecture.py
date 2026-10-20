#!/usr/bin/env python3
"""Renders the Mermaid diagrams in docs/architecture/*.md for review.

Each Markdown file becomes docs/architecture/<name>.html, a standalone page
that draws every ```mermaid block with Mermaid from cdn.jsdelivr.net. The
HTML is a build output and is not committed.

With --check, every block is also rendered with the Mermaid CLI (mmdc on
PATH, or MMDC pointing at it), and the script fails if any block does not
render. CI runs it that way.
"""
import html
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MERMAID_VERSION = "11.4.1"
ROOT = Path(__file__).resolve().parent.parent
DIAGRAMS = ROOT / "docs" / "architecture"
BLOCK = re.compile(r"^```mermaid\n(.*?)^```", re.MULTILINE | re.DOTALL)

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #1d1d1f; background: #fff; }}
  @media (prefers-color-scheme: dark) {{ body {{ color: #f5f5f7; background: #1d1d1f; }} }}
  pre.mermaid {{ background: transparent; }}
</style>
</head>
<body>
<h1>{title}</h1>
<p>Rendered from {source}.</p>
{blocks}
<script type="module">
  import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@{version}/dist/mermaid.esm.min.mjs";
  const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  mermaid.initialize({{ startOnLoad: true, theme: dark ? "dark" : "default" }});
</script>
</body>
</html>
"""


def render_page(markdown: Path, blocks: list[str]) -> None:
    title = next((line[2:].strip() for line in markdown.read_text().splitlines() if line.startswith("# ")), markdown.stem)
    body = "\n".join(f'<pre class="mermaid">\n{html.escape(block)}</pre>' for block in blocks)
    page = PAGE.format(title=html.escape(title), source=html.escape(str(markdown.relative_to(ROOT))),
                       blocks=body, version=MERMAID_VERSION)
    markdown.with_suffix(".html").write_text(page)


def check(markdown: Path, blocks: list[str]) -> bool:
    mmdc = os.environ.get("MMDC", "mmdc")
    ok = True
    with tempfile.TemporaryDirectory() as work:
        config = Path(work) / "puppeteer.json"
        # GitHub's Ubuntu runners do not allow Chromium's sandbox.
        config.write_text(json.dumps({"args": ["--no-sandbox"]}))
        for index, block in enumerate(blocks, start=1):
            source = Path(work) / f"block{index}.mmd"
            source.write_text(block)
            result = subprocess.run([mmdc, "-q", "-p", str(config), "-i", str(source), "-o", str(source.with_suffix(".svg"))],
                                    capture_output=True, text=True)
            if result.returncode != 0:
                ok = False
                print(f"{markdown.relative_to(ROOT)}: Mermaid block {index} does not render:\n{result.stderr}", file=sys.stderr)
    return ok


def main() -> int:
    validate = "--check" in sys.argv[1:]
    ok = True
    for markdown in sorted(DIAGRAMS.glob("*.md")):
        blocks = BLOCK.findall(markdown.read_text())
        if not blocks:
            continue
        render_page(markdown, blocks)
        print(f"rendered {markdown.with_suffix('.html').relative_to(ROOT)} ({len(blocks)} diagram(s))")
        if validate:
            ok = check(markdown, blocks) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
