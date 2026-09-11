"""
Render docs/USER_GUIDE.md into the browsable HTML guide (docs/user_guide.html).

    python tools/build_guide_html.py                      # -> docs/user_guide.html (standalone)
    python tools/build_guide_html.py --artifact out.html  # -> body-only copy for publishing

Needs the ``markdown`` package (any environment: ``pip install markdown``). The page shell –
palette, typography, contents rail, search – lives next to this script in ``guide_shell.html``;
this module only converts the Markdown, decorates the two tables that carry the app's own
vocabulary (step states, material grey values) and fills the shell in.

The standalone output is a complete document meant to be opened from disk. The ``--artifact``
output leaves out ``<!doctype>``/``<html>``/``<head>``/``<body>``, which is what the Artifact
publisher expects (it supplies those itself).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHELL = Path(__file__).with_name("guide_shell.html")

# step states -> the chip classes in the shell, coloured like the app's own badges
CHIPS = {
    "not started": "chip-off",
    "partly done": "chip-run",
    "done": "chip-ok",
    "stale": "chip-stale",
    "errors": "chip-err",
    "blocked": "chip-off",
}

STANDALONE_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root { color-scheme: light dark; }
html, body { margin: 0; }
img { max-width: 100%; }
[hidden] { display: none !important; }
</style>
"""


def strip_front_matter(text: str) -> str:
    """Drop the title, standfirst and contents list: the page provides its own."""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.strip() == "---":
            return "\n".join(lines[i + 1:]).lstrip("\n")
    return text


def decorate_table(table: str) -> str:
    """Give the step-state table coloured dots and the material table real grey swatches."""
    if "Badge" in table and "Meaning" in table:
        def chip(m):
            word = m.group(1)
            cls = CHIPS.get(word.lower())
            if not cls:
                return m.group(0)
            return f'<td><span class="chip {cls}"><strong>{word}</strong></span></td>'
        table = re.sub(r"<td><strong>([A-Za-z ]+)</strong></td>", chip, table)
    if "Grey value" in table:
        def swatch(m):
            v = int(m.group(1))
            return (f'<td><span class="swatch" style="--sw: rgb({v},{v},{v})" aria-hidden="true"></span>'
                    f"<code>{v}</code></td>")
        table = re.sub(r"<td><code>(\d{1,3})</code></td>", swatch, table)
    return table


def render(md_path: Path) -> str:
    try:
        import markdown
    except ImportError:
        sys.exit("This script needs the 'markdown' package: pip install markdown")
    html = markdown.markdown(
        strip_front_matter(md_path.read_text(encoding="utf-8")),
        extensions=["tables", "fenced_code", "toc", "sane_lists", "attr_list"],
        extension_configs={"toc": {"permalink": False}},
    )
    html = re.sub(r"<table>.*?</table>", lambda m: decorate_table(m.group(0)), html, flags=re.S)
    return re.sub(r"<table>.*?</table>",
                  lambda m: '<div class="table-wrap">' + m.group(0) + "</div>", html, flags=re.S)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--md", type=Path, default=REPO / "docs" / "USER_GUIDE.md", help="source Markdown")
    ap.add_argument("--out", type=Path, default=REPO / "docs" / "user_guide.html", help="standalone HTML page")
    ap.add_argument("--artifact", type=Path, default=None, help="also write a body-only copy for publishing")
    args = ap.parse_args(argv)

    shell = SHELL.read_text(encoding="utf-8")
    for marker in ("<!--BODY-->", "<!--CONTENT-->"):
        if marker not in shell:
            sys.exit(f"{SHELL.name} has no {marker} marker")
    head, body = shell.split("<!--BODY-->", 1)
    body = body.replace("<!--CONTENT-->", render(args.md))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(STANDALONE_HEAD + head + "</head>\n<body>\n" + body + "\n</body>\n</html>\n",
                        encoding="utf-8")
    print(f"wrote {args.out}")
    if args.artifact:
        args.artifact.parent.mkdir(parents=True, exist_ok=True)
        args.artifact.write_text(head + body, encoding="utf-8")
        print(f"wrote {args.artifact}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
